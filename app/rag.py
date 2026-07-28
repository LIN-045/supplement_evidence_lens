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
from sentence_transformers import SentenceTransformer

from retrieval import (
    ELASTICSEARCH_URL,
    MODEL_NAME as EMBEDDING_MODEL_NAME,
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
- For an ingredient question, first search the named ingredient.
- If a specific form is not found, also search its broader base ingredient.
- For a compound claim, search each claim separately while retaining the
  ingredient name. Do not search only the health concern when an ingredient is
  specified.
- Stop searching once the available evidence is sufficient.

Answer rules:
- Cite supporting excerpts with their provided reference numbers, such as [1]
  or [1, 2].
- Do not add factual claims unsupported by the returned excerpts.
- Distinguish regulatory claims from general health information.
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


def build_sources(
    references: list[dict[str, Any]],
    answer: str,
) -> str:
    cited_numbers = {
        int(number)
        for citation in re.findall(r"\[([\d,\s]+)\]", answer)
        for number in re.findall(r"\d+", citation)
    }

    lines = ["Sources:"]

    for number, result in enumerate(references, start=1):
        if number not in cited_numbers:
            continue

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
    openai_client: OpenAI,
) -> str:
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
            return f"{answer}\n\n{build_sources(references, answer)}"

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
                print(
                    f"Search {search_calls}/{MAX_SEARCH_CALLS}: {query}"
                )
                results = search(
                    elasticsearch_client,
                    embedding_model,
                    query,
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
    openai_client = OpenAI()

    answer = answer_question(
        arguments.question,
        elasticsearch_client,
        embedding_model,

        openai_client,
    )
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
