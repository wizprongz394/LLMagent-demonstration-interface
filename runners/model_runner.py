import time
from datetime import datetime
from typing import Dict, Any, List

from langchain_ollama import ChatOllama

from enhanced_react_agent import EnhancedReActAgent, AgentConfig


def run_model_on_query(
    model_config: Dict[str, Any],
    query_obj: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs a single LLM model on a single GIS query using the EnhancedReActAgent,
    logs intent classification, and returns structured metrics.
    """

    # -----------------------------
    # Model configuration
    # -----------------------------
    model_name = model_config["name"]
    model_id = model_config["model_id"]
    temperature = model_config.get("temperature", 0)
    max_iterations = model_config.get("max_iterations", 15)

    # -----------------------------
    # Query information
    # -----------------------------
    query_id = query_obj["id"]
    query_text = query_obj["query"]

    # -----------------------------
    # Tracking variables
    # -----------------------------
    start_time = time.time()

    iterations_used = 0
    tool_calls = 0
    tools_used = set()

    classified_intent = "unknown"
    success = True
    error_message = None
    iteration_details: List[Dict[str, Any]] = []

    # -----------------------------
    # Initialize Agent
    # -----------------------------
    agent_config = AgentConfig(
        model_name=model_id,
        temperature=temperature,
        max_iterations=max_iterations,
        enable_memory=False
    )

    agent = EnhancedReActAgent(config=agent_config)

    # -----------------------------
    # Run query and capture events
    # -----------------------------
    try:
        for step in agent.run_query(query_text):

            step_type = step.get("type")

            # Capture intent classification
            if step_type == "intent_classification":
                classified_intent = step.get("intent", "unknown")

            # Track iteration progress
            elif step_type == "iteration_update":
                iterations_used = step.get("current", iterations_used)

            # Capture final answer & reasoning trace
            elif step_type == "final_answer":
                iteration_details = step.get("details", [])

    except Exception as e:
        success = False
        error_message = str(e)
        iteration_details = []

    # -----------------------------
    # Analyze tool usage
    # -----------------------------
    for detail in iteration_details:
        action = detail.get("action")
        if action:
            tool_calls += 1
            tools_used.add(action)

    # -----------------------------
    # Stop timer
    # -----------------------------
    response_time = round(time.time() - start_time, 3)

    # -----------------------------
    # Build structured result
    # -----------------------------
    result = {
        "model_name": model_name,
        "model_id": model_id,

        "query_id": query_id,
        "query_text": query_text,

        # NEW: intent classification
        "classified_intent": classified_intent,

        "response_time_sec": response_time,
        "iterations_used": iterations_used,
        "tool_calls": tool_calls,
        "tools_used": list(tools_used),

        "success": success,
        "error": error_message,

        "timestamp": datetime.utcnow().isoformat(),

        # Raw trace for UI / deeper analysis
        "iteration_details": iteration_details
    }

    return result
