"""Answer questions with one fixed retrieval step."""

import json
import re
from typing import Any

from openai import OpenAI

from app.retrieval import EvidenceRetriever


LLM_MODEL_NAME = "gpt-5.4-mini"
BASELINE_RAG_VERSION = "baseline_rag_v1"
MINIMUM_CHUNK_OVERLAP = 50
CITATION_PATTERN = re.compile(r"\[([\d,\s\-–—]+)\]")

ANSWER_RULES = """
- Start with a direct conclusion whose certainty matches the evidence.
- Clearly distinguish limited, mixed, indirect, and unsupported evidence.
- Cite supporting excerpts with reference numbers such as [1] or [1, 2].
- Do not add factual claims unsupported by the supplied excerpts.
- Evidence role describes a source's purpose, not its quality, authority,
  relevance, or priority. Use it only to avoid confusing regulatory claims,
  clinical summaries, nutrient reference values, and consumer guidance.
- Do not treat "studied for" as proof of effectiveness.
- Preserve distinctions such as "studied for", "may help", and "authorised
  claim".
- Distinguish regulatory claims from general health information.
- Separate jurisdictions when their findings disagree or serve different roles.
- For dose questions, report only source-stated dose ranges together with their
  use, population, jurisdiction, and conditions.
- Do not turn source-stated doses into personalised recommendations.
- Do not imply that the upper end of a dose range is more effective, optimal,
  safer, or an appropriate starting dose.
- Do not infer a dose-response relationship unless a source explicitly states
  one.
- Do not present doses mentioned only in references as official recommendations.
- If the excerpts do not answer the question, say so explicitly.
- Do not make definitive legal judgments or provide personalised medical advice.
- Keep the answer clear and concise.
"""

BASELINE_INSTRUCTIONS = f"""
Answer the dietary supplement question using only the supplied official-source
excerpts.

Answer rules:
{ANSWER_RULES}
"""


class BaseRAG:
    """Answer with one retrieval call followed by one LLM call."""

    rag_version = BASELINE_RAG_VERSION

    def __init__(
        self,
        retriever: EvidenceRetriever,
        openai_client: OpenAI,
        *,
        model_name: str = LLM_MODEL_NAME,
    ) -> None:
        self.retriever = retriever
        self.openai_client = openai_client
        self.model_name = model_name

    @staticmethod
    def _prompt_contexts(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        contexts = []

        for reference, result in enumerate(results, start=1):
            document = result["_source"]
            contexts.append(
                {
                    "reference": reference,
                    "title": document["title"],
                    "source": document["source"],
                    "jurisdiction": document["jurisdiction"],
                    "evidence_role": document["evidence_role"],
                    "source_url": document["source_url"],
                    "excerpt": document["content"],
                }
            )

        return contexts

    @staticmethod
    def _merge_overlapping_content(
        existing_content: str,
        new_content: str,
    ) -> str:
        """Merge two excerpts without repeating their shared overlap."""

        if not new_content or new_content in existing_content:
            return existing_content
        if not existing_content or existing_content in new_content:
            return new_content

        maximum_overlap = min(
            len(existing_content),
            len(new_content),
        )

        for overlap in range(
            maximum_overlap,
            MINIMUM_CHUNK_OVERLAP - 1,
            -1,
        ):
            if existing_content[-overlap:] == new_content[:overlap]:
                return existing_content + new_content[overlap:]
            if new_content[-overlap:] == existing_content[:overlap]:
                return new_content + existing_content[overlap:]

        return f"{existing_content}\n\n{new_content}"

    @staticmethod
    def _citation_numbers(citation: str) -> list[int]:
        """Expand comma-separated citation numbers and numeric ranges."""

        numbers = []

        for part in citation.split(","):
            part = part.strip()
            if part.isdigit():
                numbers.append(int(part))
                continue

            range_match = re.fullmatch(
                r"(\d+)\s*[-–—]\s*(\d+)",
                part,
            )
            if range_match is None:
                continue

            start, end = map(int, range_match.groups())
            if start <= end:
                numbers.extend(range(start, end + 1))

        return numbers

    @staticmethod
    def _compact_citations(
        answer: str,
        references: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        cited_numbers: list[int] = []

        for citation in CITATION_PATTERN.findall(answer):
            for number in BaseRAG._citation_numbers(citation):
                if (
                    1 <= number <= len(references)
                    and number not in cited_numbers
                ):
                    cited_numbers.append(number)

        number_map: dict[int, int] = {}
        cited_references: list[dict[str, Any]] = []
        source_document_numbers: dict[str, int] = {}

        for old_number in cited_numbers:
            result = references[old_number - 1]
            document = result["_source"]
            source_document_id = document.get(
                "source_document_id",
                result["_id"],
            )

            if source_document_id not in source_document_numbers:
                new_number = len(cited_references) + 1
                source_document_numbers[
                    source_document_id
                ] = new_number
                cited_references.append(
                    {
                        **result,
                        "_source": {**document},
                    }
                )
            else:
                new_number = source_document_numbers[
                    source_document_id
                ]
                existing_document = cited_references[
                    new_number - 1
                ]["_source"]
                new_content = document.get("content", "")
                existing_document["content"] = (
                    BaseRAG._merge_overlapping_content(
                        existing_document.get("content", ""),
                        new_content,
                    )
                )

            number_map[old_number] = new_number

        def replace_citation(match: re.Match[str]) -> str:
            old_numbers = BaseRAG._citation_numbers(
                match.group(1)
            )
            new_numbers = []

            for number in old_numbers:
                mapped_number = number_map.get(number)

                if (
                    mapped_number is not None
                    and mapped_number not in new_numbers
                ):
                    new_numbers.append(mapped_number)

            if not new_numbers:
                return ""

            return f"[{', '.join(map(str, new_numbers))}]"

        compacted_answer = CITATION_PATTERN.sub(
            replace_citation,
            answer,
        )
        compacted_answer = re.sub(
            r"\s+([.,;:!?])",
            r"\1",
            compacted_answer,
        )
        return compacted_answer, cited_references

    @staticmethod
    def _build_contexts(
        references: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        contexts = []

        for reference, result in enumerate(references, start=1):
            document = result["_source"]
            contexts.append(
                {
                    "reference": reference,
                    "document_id": result["_id"],
                    "source_document_id": document.get(
                        "source_document_id",
                        result["_id"],
                    ),
                    "title": document["title"],
                    "source": document["source"],
                    "jurisdiction": document["jurisdiction"],
                    "evidence_role": document["evidence_role"],
                    "source_url": document["source_url"],
                    "content": document["content"],
                }
            )

        return contexts

    def _build_result(
        self,
        *,
        question: str,
        answer: str,
        references: list[dict[str, Any]],
        search_queries: list[str],
    ) -> dict[str, Any]:
        answer, cited_references = self._compact_citations(
            answer,
            references,
        )

        return {
            "question": question,
            "answer": answer,
            "search_queries": search_queries,
            "contexts": self._build_contexts(cited_references),
        }

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:
        """Answer using the original question as the only search query."""

        results = self.retriever.search(question)
        prompt = json.dumps(
            {
                "question": question,
                "contexts": self._prompt_contexts(results),
            },
            ensure_ascii=False,
        )

        response = self.openai_client.responses.create(
            model=self.model_name,
            instructions=BASELINE_INSTRUCTIONS,
            input=prompt,
        )

        return self._build_result(
            question=question,
            answer=response.output_text,
            references=results,
            search_queries=[question],
        )
