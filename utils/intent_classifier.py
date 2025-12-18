from typing import Dict
import json

from langchain_ollama import ChatOllama


INTENT_LABELS = [
    "weather",
    "road",
    "risk",
    "construction",
    "comparison",
    "geospatial",
    "unknown"
]


def classify_query_intent(
    query: str,
    model_name: str,
    temperature: float = 0.0
) -> Dict[str, str]:
    """
    Uses an LLM to classify the intent of a GIS query.
    Returns a strict intent label.
    """

    llm = ChatOllama(
        model=model_name,
        temperature=temperature
    )

    system_prompt = (
        "You are an expert GIS query classifier.\n"
        "Classify the user's query into exactly ONE of the following labels:\n\n"
        f"{', '.join(INTENT_LABELS)}\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON\n"
        "- Do NOT explain your reasoning\n"
        "- If uncertain, choose 'unknown'\n\n"
        "JSON format:\n"
        '{ "intent": "<label>" }'
    )

    user_prompt = f"Query: {query}"

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    try:
        parsed = json.loads(response.content)
        intent = parsed.get("intent", "unknown")

        if intent not in INTENT_LABELS:
            intent = "unknown"

    except Exception:
        intent = "unknown"

    return {
        "intent": intent
    }
