"""Generate retrieval evaluation questions from processed documents.

Run from the project root:

    uv run python evaluation/generate_questions.py
"""

import json
import os
import random
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# Configuration

INPUT_PATH = Path("data/processed/document_chunks.jsonl")
OUTPUT_PATH = Path("data/evaluation/questions.jsonl")

MODEL_NAME = "gpt-5.4-mini"
RANDOM_SEED = 42

QUESTIONS_PER_SOURCE = 25

SOURCE_NAMES = {
    "eu_health_claims_register": "EU Register",
    "health_canada_nhpid": "Health Canada",
    "nih_ods": "NIH ODS",
}

QUESTION_GENERATION_INSTRUCTIONS = """
Generate one realistic question that an ordinary consumer might ask a dietary
supplement evidence assistant after seeing a product label, advertisement, or
supplement ingredient.

Requirements:
- The question must be answerable from the supplied excerpt.
- Write in clear, everyday English and avoid database or regulatory jargon.
- Make the question understandable without showing the excerpt.
- Name the relevant ingredient, supplement, or product when needed.
- Use the ingredient name a consumer would recognise. Omit long chemical,
  botanical, strain, or product identifiers unless they are necessary to
  distinguish the substance. The answer may clarify the exact form or strain
  covered by the source.
- Prefer practical questions about whether a claim is supported or allowed,
  what an ingredient may do, dose information, safety, restrictions, or who
  the information applies to.
- Regulatory questions may mention the EU, Health Canada, or NIH, but avoid
  phrases such as "What is the regulatory status of the health claim regarding".
- Do not ask about authors, publication dates, section names, database fields,
  or citation numbers.
- Do not assume that a marketing claim is authorised or scientifically proven.
- Return only the question.
"""

# Data loading

def load_documents(path: Path) -> list[dict[str, Any]]:
    documents = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            documents.append(json.loads(line))

    return documents


# Sampling

def is_eligible_for_question_generation(
    document: dict[str, Any],
) -> bool:
    section_title = document.get("section_title", "").lower()

    excluded_sections = {
        "date",
        "references cited",
        "references reviewed",
    }

    if section_title in excluded_sections:
        return False

    if document["source"] == "eu_health_claims_register":
        return bool(document.get("claim_text"))

    return len(document.get("content", "").strip()) >= 300


def parent_document_id(document: dict[str, Any]) -> str:
    if document["source"] == "health_canada_nhpid":
        return document["monograph_id"]

    if document["source"] == "nih_ods":
        return document["fact_sheet_id"]

    return document["source_document_id"]


def sample_documents(
    documents: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(RANDOM_SEED)
    samples = {}

    for source in SOURCE_NAMES:
        candidates = [
            document
            for document in documents
            if document["source"] == source
            and is_eligible_for_question_generation(document)
        ]
        rng.shuffle(candidates)

        selected = []
        selected_parents = set()

        for document in candidates:
            parent_id = parent_document_id(document)

            if parent_id in selected_parents:
                continue

            selected.append(document)
            selected_parents.add(parent_id)

            if len(selected) == QUESTIONS_PER_SOURCE:
                break

        if len(selected) < QUESTIONS_PER_SOURCE:
            raise ValueError(
                f"Not enough eligible documents for {source}: "
                f"{len(selected)}"
            )

        samples[source] = selected

    return samples

def generate_question(
    client: OpenAI,
    document: dict[str, Any],
) -> str:
    source_name = SOURCE_NAMES[document["source"]]

    prompt = f"""
Source: {source_name}
Title: {document["title"]}

Excerpt:
{document["content"]}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=QUESTION_GENERATION_INSTRUCTIONS,
        input=prompt,
    )

    question = response.output_text.strip()

    if not question:
        raise ValueError(
            f"No question generated for {document['document_id']}"
        )

    return question

def generate_source_questions(
    client: OpenAI,
    samples: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    questions = []
    question_number = 1

    for source, documents in samples.items():
        for document in documents:
            question = generate_question(client, document)

            questions.append(
                {
                    "question_id": f"source-{question_number:03d}",
                    "question": question,
                    "question_group": "source_generated",
                    "seed_document_id": document["document_id"],
                    "source": source,
                    "title": document["title"],
                    "source_url": document["source_url"],
                }
            )

            print(
                f"[{question_number}/{QUESTIONS_PER_SOURCE * len(SOURCE_NAMES)}] "
                f"{question}"
            )
            question_number += 1

    return questions


def write_questions(
    questions: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for question in questions:
            file.write(
                json.dumps(question, ensure_ascii=False) + "\n"
            )

def main() -> None:
    load_dotenv()

    documents = load_documents(INPUT_PATH)
    samples = sample_documents(documents)

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    questions = generate_source_questions(client, samples)
    write_questions(questions, OUTPUT_PATH)

    print(f"Wrote {len(questions)} questions to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
