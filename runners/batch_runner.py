import json
import os
from datetime import datetime
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from runners.model_runner import run_model_on_query


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(DATA_DIR, "results")


def load_models_config() -> List[Dict[str, Any]]:
    """Load enabled models from models.yaml"""
    models_path = os.path.join(CONFIG_DIR, "models.yaml")

    with open(models_path, "r") as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    enabled_models = [m for m in models if m.get("enabled", False)]

    return enabled_models


def load_queries() -> List[Dict[str, Any]]:
    """Load dummy6 queries"""
    queries_path = os.path.join(DATA_DIR, "dummy6_queries.json")

    with open(queries_path, "r") as f:
        queries = json.load(f)

    return queries


def save_results(results: List[Dict[str, Any]]) -> str:
    """Save experiment results to disk"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    return filepath


def run_batch_experiment() -> List[Dict[str, Any]]:
    """
    Runs all enabled models on all dummy queries
    and returns a list of structured experiment results.
    """

    models = load_models_config()
    queries = load_queries()

    all_results: List[Dict[str, Any]] = []

    for model in models:
        model_name = model["name"]

        for query in queries:
            print(f"[RUNNING] Model={model_name} | Query={query['id']}")

            result = run_model_on_query(
                model_config=model,
                query_obj=query
            )

            all_results.append(result)

    results_file = save_results(all_results)
    print(f"\n✅ Experiment completed.")
    print(f"📁 Results saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    run_batch_experiment()
