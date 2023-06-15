from datasets import load_dataset
import json
import os


def load_from_path(path):
    is_yes_no = True
    with open(path, "r") as f:
        key2promt_completions = json.load(f)

    for key, prompt_completions in key2promt_completions.items():
        tmp_file = f"tmp_{key}.jsonl"
        with open(tmp_file, "w") as f:
            for idx, prompt_completion in enumerate(prompt_completions):
                d = {
                    "input": prompt_completion["prompt"],
                    "output": prompt_completion["completion"],
                    "provenance": f"path={path}, key={key}, idx={idx}",
                }
                if (
                    "yes" not in prompt_completion["completion"].lower()
                    and "no" not in prompt_completion["completion"].lower()
                ):
                    is_yes_no = False
                f.write(json.dumps(d) + "\n")
    f_dict = {
        "train": f"tmp_train.jsonl",
        "test": f"tmp_eval.jsonl",
    }
    dataset = load_dataset("json", data_files=f_dict)
    for f in f_dict.values():
        os.remove(f)
    return_dict = {
        "dataset": dataset,
        "is_yes_no": is_yes_no,
        "key2promt_completions": key2promt_completions,
    }
    if is_yes_no:
        return_dict["test_labels"] = [
            "yes" in d["completion"].lower() for d in key2promt_completions["eval"]
        ]
    return return_dict


if __name__ == "__main__":
    # test_path = "data/0105proposer_all_data.json"
    test_path = "../descriptive_clustering/scratch/balanced_verifier_prompt_completion_data.json"
    dataset = load_from_path(test_path)
    print(dataset)
