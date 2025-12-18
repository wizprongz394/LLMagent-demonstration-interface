import json
from typing import Dict, List
from collections import defaultdict


def score_experiment(results: List[Dict]) -> Dict[str, Dict]:
    """
    Converts raw experiment results into per-model scores.
    """

    model_stats = defaultdict(lambda: {
        "runs": 0,
        "success": 0,
        "avg_response_time": 0.0,
        "avg_iterations": 0.0,
        "intent_known": 0,
        "final_score": 0.0
    })

    # -----------------------------
    # Aggregate raw metrics
    # -----------------------------
    for r in results:
        model = r["model_name"]
        model_stats[model]["runs"] += 1

        if r["success"]:
            model_stats[model]["success"] += 1

        model_stats[model]["avg_response_time"] += r["response_time_sec"]
        model_stats[model]["avg_iterations"] += r["iterations_used"]

        if r["classified_intent"] != "unknown":
            model_stats[model]["intent_known"] += 1

    # -----------------------------
    # Normalize + score
    # -----------------------------
    for model, stats in model_stats.items():
        runs = stats["runs"]

        stats["avg_response_time"] /= runs
        stats["avg_iterations"] /= runs

        success_rate = stats["success"] / runs
        intent_rate = stats["intent_known"] / runs

        # ---- Scoring logic (simple, interpretable) ----
        score = 0.0
        score += success_rate * 0.4
        score += intent_rate * 0.3
        score += max(0, 1 - stats["avg_response_time"] / 120) * 0.2
        score += max(0, 1 - stats["avg_iterations"] / 10) * 0.1

        stats["final_score"] = round(score, 3)

    return dict(model_stats)


def load_and_score(path: str) -> Dict[str, Dict]:
    with open(path, "r") as f:
        results = json.load(f)

    return score_experiment(results)


if __name__ == "__main__":
    import sys
    scored = load_and_score(sys.argv[1])
    print(json.dumps(scored, indent=2))
