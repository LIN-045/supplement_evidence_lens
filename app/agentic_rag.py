"""Answer questions with agent-directed retrieval."""

import json
from typing import Any

from app.base_rag import ANSWER_RULES, BaseRAG


MAX_SEARCH_CALLS = 4
MAX_AGENT_TURNS = 8
AGENTIC_RAG_VERSION = "agentic_rag_v4"

AGENT_INSTRUCTIONS = f"""
You answer dietary supplement questions using the search_official_sources tool
and only the official source excerpts it returns.

Retrieval rules:
- Always search before answering.
- Keep each search query short and focused.
- For an ingredient question, first search the named ingredient.
- For a broad effectiveness question, first search the ingredient without
  assuming one specific intended effect.
- If a specific form is not found, also search its broader base ingredient.
- For a compound question, use separate focused searches for distinct effects.
- For a dose question, include the ingredient, dose, use, and population when
  relevant.
- If the results do not contain the requested evidence, refine the query while
  searches remain.
- Check whether the retrieved excerpts cover every important part of the
  question before answering.
- Stop searching once the available evidence is sufficient.

Answer rules:
{ANSWER_RULES}
"""

SEARCH_TOOL = {
    "type": "function",
    "name": "search_official_sources",
    "description": (
        "Search indexed official supplement evidence using hybrid keyword "
        "and semantic retrieval."
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


class AgenticRAG(BaseRAG):
    """Allow the LLM to select and refine multiple search queries."""

    rag_version = AGENTIC_RAG_VERSION

    @staticmethod
    def _normalise_query(query: str) -> str:
        return " ".join(query.casefold().split())

    @staticmethod
    def _format_search_results(
        results: list[dict[str, Any]],
        references: list[dict[str, Any]],
        reference_numbers: dict[str, int],
    ) -> str:
        documents = []

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
                    "evidence_role": document["evidence_role"],
                    "url": document["source_url"],
                    "excerpt": document["content"],
                }
            )

        return json.dumps(
            {"results": documents},
            ensure_ascii=False,
        )

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:
        """Answer using LLM-directed query selection and refinement."""

        input_items: list[Any] = [
            {
                "role": "user",
                "content": question,
            }
        ]
        search_cache: dict[str, str] = {}
        references: list[dict[str, Any]] = []
        reference_numbers: dict[str, int] = {}
        search_queries: list[str] = []
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

            response = self.openai_client.responses.create(
                model=self.model_name,
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
                return self._build_result(
                    question=question,
                    answer=response.output_text,
                    references=references,
                    search_queries=search_queries,
                )

            input_items.extend(response.output)

            for tool_call in tool_calls:
                arguments = json.loads(tool_call.arguments)
                query = arguments["query"].strip()
                cache_key = self._normalise_query(query)

                if not query:
                    tool_output = json.dumps(
                        {"error": "Search query must not be empty"}
                    )
                elif cache_key in search_cache:
                    tool_output = search_cache[cache_key]
                elif search_calls >= MAX_SEARCH_CALLS:
                    tool_output = json.dumps(
                        {
                            "error": "Search limit reached",
                            "instruction": (
                                "Answer using the existing results."
                            ),
                        }
                    )
                else:
                    search_calls += 1
                    search_queries.append(query)
                    results = self.retriever.search(query)
                    tool_output = self._format_search_results(
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

        raise RuntimeError(
            "Agent did not produce an answer within the turn limit"
        )
