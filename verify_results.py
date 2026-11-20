"""Verify the release measurements from the per-request datasets."""

import json
import math
from pathlib import Path
from statistics import median

from paired_bench import compare, consistency

RESULTS = Path(__file__).resolve().parent / "results"
MODEL_SHA256 = "392e8d466d56100ada00eb82031fb854297fc9e389b7d303eba3af114e87bce2"


def require(condition, description):
    if not condition:
        raise ValueError(description)


def verify(filename, rounds, budget_mib, pool_bytes):
    document = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
    require(document["weight_sha256"] == MODEL_SHA256, f"{filename}: unexpected weights")
    trials = document["trials"]
    expected = {(round_number, mode) for round_number in range(1, rounds + 1)
                for mode in ("stock", "coalesced")}
    keys = [(trial["round"], trial["mode"]) for trial in trials]
    require(len(keys) == len(expected) and set(keys) == expected, f"{filename}: invalid pairs")
    for trial in trials:
        require(trial["pool"]["bytes"] == pool_bytes, f"{filename}: unexpected KV pool size")
        require(trial["pool"]["dtype"] == "bfloat16", f"{filename}: unexpected KV dtype")
        require(trial["mlx_memory_guideline_bytes"] == budget_mib * 2**20,
                f"{filename}: unexpected MLX guideline")
        require(trial["mlx_peak_allocated_bytes"] <= trial["mlx_memory_guideline_bytes"],
                f"{filename}: measured peak over budget")
    require(not consistency(trials), f"{filename}: differing outputs")
    calculated = compare(trials)
    require(calculated == document["comparisons"], f"{filename}: results differ after recomputing")
    total = sum(len(rows) for trial in trials for rows in trial["cases"].values())
    print(f"{filename}: {total} requests across {rounds} valid pairs")
    return trials, {entry["scenario"]: entry for entry in calculated}


def percentile95(values):
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main():
    trials, parallel = verify("adaptive-parallel.json", 4, 6144, 939638784)
    verify("adaptive-serial.json", 2, 3072, 352436224)
    _, chat = verify("adaptive-chat-clean.json", 3, 4096, 939638784)
    case = "parallel_chat_forks"
    require(all(len(trial["cases"][case]) == 12 for trial in trials), "unexpected parallel request count")
    result = parallel[case]
    require(result["wins"] == 4 and round(result["paired_median_reduction_pct"], 1) == 10.7,
            "headline parallel result differs")
    print(f"{case}: {result['wins']}/4 wins, {result['paired_median_reduction_pct']:.2f}% paired median reduction")
    for mode in ("stock", "coalesced"):
        times = [row["first_text_ms"] for trial in trials if trial["mode"] == mode
                 for row in trial["cases"][case]]
        print(f"{mode}: P50 {median(times):.1f} ms, P95 {percentile95(times):.1f} ms")

    slow = chat["chat_history"]
    require(slow["wins"] == 2, "unexpected continuous-chat win count")
    for pair in slow["paired_trials"]:
        print(f"chat_history round {pair['round']}: "
              f"{pair['stock_ms']:.1f} -> {pair['candidate_ms']:.1f} ms")


if __name__ == "__main__":
    main()
