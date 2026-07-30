"""Shared JSON Schema output helper for evaluation judges."""

import json
from typing import Any

from openai import OpenAI


def generate_structured_output(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    input_text: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Call an evaluation model and parse its strict JSON Schema output."""

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )

    result = json.loads(response.output_text)

    if not isinstance(result, dict):
        raise TypeError(
            f"Expected {schema_name} output to be a JSON object"
        )

    return result
