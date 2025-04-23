import json
from pathlib import Path


def get_overall_average(eval_data: dict) -> float:
    dset_scores = []

    for category in eval_data["categories"]:
        for dataset in category["datasets"]:
            dset_scores.append(dataset["average_score"])

    if not dset_scores:
        raise ValueError("No datasets found in the evaluation data.")

    return sum(dset_scores) / len(dset_scores)


def get_valid_average(eval_data: dict) -> float:
    scores = []

    for category in eval_data["categories"]:
        for dataset in category["datasets"]:
            for result in dataset["results"]:
                for completion in result["completions"]:
                    if completion["model_answer"] is not None:
                        scores.append(completion["score"])

    if not scores:
        raise ValueError("No valid scores found.")

    return sum(scores) / len(scores)


def score(model_name, train_name, eval_names, valid_only=False):
    for eval_name in eval_names:
        trained_results_path = Path(f"inter_{train_name}_to_{eval_name}/{model_name}_to_{eval_name}.json")
        original_results_path = Path(f"inter_original_to_{eval_name}/original_to_{eval_name}.json")

        trained_results = json.loads(trained_results_path.read_text())
        original_results = json.loads(original_results_path.read_text())

        if valid_only:
            trained_overall = get_valid_average(trained_results)
            original_overall = get_valid_average(original_results)
        else:
            trained_overall = get_overall_average(trained_results)
            original_overall = get_overall_average(original_results)

        print(f"Original -> Trained ({train_name} train, {eval_name} eval)")
        print(f"{original_overall:.3f} -> {trained_overall:.3f} [{trained_overall/original_overall:.2f}x]")


if __name__ == "__main__":
    score("algorithmic_qwen_3b_500", "algorithmic", ["algebra", "arithmetic", "geometry"], valid_only=True)
