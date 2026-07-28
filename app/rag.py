"""Answer supplement questions with agent-directed retrieval.

Run from the project root:

    uv run python app/rag.py "What are the risks of high-dose zinc?"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)
if __package__:
    from .retrieval import (
        ELASTICSEARCH_URL,
        MODEL_NAME as EMBEDDING_MODEL_NAME,
        RERANKER_MODEL_NAME,
        search,
    )
else:
    from retrieval import (
        ELASTICSEARCH_URL,
        MODEL_NAME as EMBEDDING_MODEL_NAME,
        RERANKER_MODEL_NAME,
        search,
    )


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LLM_MODEL_NAME = "gpt-5.4-mini"
MAX_SEARCH_CALLS = 4
MAX_AGENT_TURNS = 8

AGENT_INSTRUCTIONS = """You answer dietary supplement questions using the
search_official_sources tool and only the official source excerpts it returns.

Retrieval rules:
- Always search before answering.
- Decide whether one or several searches are needed.
- Keep each search query short and focused. Do not include generic phrases such
  as "official sources", because the tool already searches only official data.
- For an ingredient question, first search the named ingredient.
- For a broad question such as "Does X work?", first search the ingredient
  without assuming a single intended effect, then describe only the uses covered
  by the results.
- If a specific form is not found, also search its broader base ingredient.
- For a compound claim, use a separate search for each claimed effect while
  retaining the ingredient name. Do not combine multiple effects in one query,
  and do not search only the health concern when an ingredient is specified.
- For a dose question, search the ingredient together with "dose" and the
  stated use or population. If the results do not contain an explicit dose or
  dosage section, refine the query before answering when searches remain.
- Before answering, check whether the retrieved excerpts directly cover every
  part of the user's question. If an important part is unsupported and searches
  remain, run another focused search for that missing part. Do not claim that
  the sources lack information until you have run a focused search for it.
- Stop searching once the available evidence is sufficient.

Answer rules:
- Cite supporting excerpts with their provided reference numbers, such as [1]
  or [1, 2].
- Do not add factual claims unsupported by the returned excerpts.
- Do not treat "studied for" as proof of effectiveness. Preserve distinctions
  such as "studied for", "may help", and "authorised claim".
- Distinguish regulatory claims from general health information.
- For dose questions, report only source-stated dose ranges and their use,
  population, jurisdiction, and conditions. Do not turn them into a personalised
  recommendation about what the user should buy or take.
- A source-stated dose range does not mean that its upper end is more effective,
  optimal, safer, or an appropriate starting dose. Do not infer a dose-response
  relationship unless a retrieved source explicitly states one.
- Do not present doses merely mentioned in study references as an official dose
  recommendation or recognised dose range.
- If the excerpts do not answer the question, say so explicitly.
- Do not make definitive legal judgments or provide personalised medical advice.
- Keep the answer clear and concise.
"""

SEARCH_TOOL = {
    "type": "function",
    "name": "search_official_sources",
    "description": (
        "Search the indexed EU, Health Canada, and NIH official supplement "
        "sources using hybrid keyword and semantic retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused English search query.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


# Search result handling

def normalise_query(query: str) -> str:
    return " ".join(query.casefold().split())


def format_search_results(
    results: list[dict[str, Any]],
    references: list[dict[str, Any]],
    reference_numbers: dict[str, int],
) -> str:
    documents: list[dict[str, Any]] = []

    for result in results:
        document_id = result["_id"]

        if document_id not in reference_numbers:
            reference_numbers[document_id] = len(references) + 1
            references.append(result)

        document = result["_source"]
        documents.append(
            {
                "reference": reference_numbers[document_id],
                "title": document["title"],
                "source": document["source"],
                "jurisdiction": document["jurisdiction"],
                "url": document["source_url"],
                "excerpt": document["content"],
            }
        )

    return json.dumps(
        {"results": documents},
        ensure_ascii=False,
    )


def compact_citations(
    answer: str,
    references: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    cited_numbers: list[int] = []

    for citation in re.findall(r"\[([\d,\s]+)\]", answer):
        for number_text in re.findall(r"\d+", citation):
            number = int(number_text)

            if (
                1 <= number <= len(references)
                and number not in cited_numbers
            ):
                cited_numbers.append(number)

    number_map = {
        old_number: new_number
        for new_number, old_number in enumerate(cited_numbers, start=1)
    }

    def replace_citation(match: re.Match[str]) -> str:
        old_numbers = [
            int(number)
            for number in re.findall(r"\d+", match.group(1))
        ]
        new_numbers = [
            number_map[number]
            for number in old_numbers
            if number in number_map
        ]
        return f"[{', '.join(map(str, new_numbers))}]"

    compacted_answer = re.sub(
        r"\[([\d,\s]+)\]",
        replace_citation,
        answer,
    )
    cited_references = [
        references[number - 1]
        for number in cited_numbers
    ]

    return compacted_answer, cited_references


def build_sources(references: list[dict[str, Any]]) -> str:
    lines = ["Sources:"]

    for number, result in enumerate(references, start=1):
        document = result["_source"]
        lines.append(
            f"[{number}] {document['title']} — {document['source_url']}"
        )

    return "\n".join(lines)


# Agentic RAG

def answer_question(
    question: str,
    elasticsearch_client: Elasticsearch,
    embedding_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    openai_client: OpenAI,
    return_trace: bool = False,
) -> str | dict[str, Any]:
    input_items: list[Any] = [
        {
            "role": "user",
            "content": question,
        }
    ]
    search_cache: dict[str, str] = {}
    references: list[dict[str, Any]] = []
    reference_numbers: dict[str, int] = {}
    search_calls = 0
    search_queries = []

    for turn in range(MAX_AGENT_TURNS):
        if turn == 0:
            tool_choice = "required"
        elif (
            search_calls >= MAX_SEARCH_CALLS
            or turn == MAX_AGENT_TURNS - 1
        ):
            tool_choice = "none"
        else:
            tool_choice = "auto"

        response = openai_client.responses.create(
            model=LLM_MODEL_NAME,
            instructions=AGENT_INSTRUCTIONS,
            input=input_items,
            tools=[SEARCH_TOOL],
            tool_choice=tool_choice,
            parallel_tool_calls=False,
        )

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            answer = response.output_text
            answer, cited_references = compact_citations(
                answer,
                references,
            )
            full_response = (
                f"{answer}\n\n"
                f"{build_sources(cited_references)}"
            )

            if not return_trace:
                return full_response

            contexts = []

            for reference_number, result in enumerate(
                cited_references,
                start=1,
            ):
                document = result["_source"]

                contexts.append(
                    {
                        "reference": reference_number,
                        "document_id": result["_id"],
                        "title": document["title"],
                        "source": document["source"],
                        "jurisdiction": document["jurisdiction"],
                        "source_url": document["source_url"],
                        "content": document["content"],
                    }
                )

            return {
                "question": question,
                "answer": answer,
                "response": full_response,
                "search_queries": search_queries,
                "contexts": contexts,
            }

        input_items.extend(response.output)

        for tool_call in tool_calls:
            arguments = json.loads(tool_call.arguments)
            query = arguments["query"].strip()
            cache_key = normalise_query(query)

            if cache_key in search_cache:
                tool_output = search_cache[cache_key]
            elif search_calls >= MAX_SEARCH_CALLS:
                tool_output = json.dumps(
                    {
                        "error": "Search limit reached",
                        "instruction": "Answer using the existing results.",
                    }
                )
            else:
                search_calls += 1
                search_queries.append(query)
                print(
                    f"Search {search_calls}/{MAX_SEARCH_CALLS}: {query}"
                )
                results = search(
                    elasticsearch_client,
                    embedding_model,
                    query,
                    reranker_model,
                )
                tool_output = format_search_results(
                    results,
                    references,
                    reference_numbers,
                )
                search_cache[cache_key] = tool_output

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": tool_output,
                }
            )

    raise RuntimeError("Agent did not produce an answer within the turn limit")

# Command-line interface

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer questions using official supplement sources."
    )
    parser.add_argument(
        "question",
        help="Dietary supplement question",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    arguments = parse_arguments()

    elasticsearch_client = Elasticsearch(ELASTICSEARCH_URL)
    if not elasticsearch_client.ping():
        raise ConnectionError(
            f"Cannot connect to Elasticsearch at {ELASTICSEARCH_URL}"
        )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
    openai_client = OpenAI()

    answer = answer_question(
        arguments.question,
        elasticsearch_client,
        embedding_model,
        reranker_model,
        openai_client,
    )
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
