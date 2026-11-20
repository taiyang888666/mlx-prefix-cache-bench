"""Run matched SGLang/MLX trials, with isolated engines and verified KV pools."""

import argparse
import hashlib
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from workloads import SCENARIOS, dossier, open_port, run_workload, send_request

MODES = ("stock", "readout", "layout", "both", "compact", "coalesced")
LONG_SCENARIOS = ("long_branches", "long_linear")
CHAT_SCENARIOS = ("chat_forks", "chat_history")
PARALLEL_SCENARIOS = ("parallel_chat_forks", "parallel_mixed_forks")
POOL_PATTERN = re.compile(
    r"MlxAttentionKVPool: (\d+) slots x (\d+) layers x (\d+) heads x (\d+) dim, "
    r"dtype=mlx.core.(\w+)"
)
READY = "The server is fired up and ready to roll!"


def wait_ready(process, log_path):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and process.poll() is None:
        if READY in log_path.read_text(errors="replace"):
            return
        time.sleep(0.25)
    raise RuntimeError(f"SGLang failed readiness; see {log_path}")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=8)


def flush(url):
    with urlopen(Request(f"{url}/flush_cache?timeout=10", method="POST"), timeout=20) as reply:
        reply.read()


def long_workload(scenario):
    if scenario == "long_branches":
        documents = [dossier(name, rows=84) for name in ("LegacyInventory", "RenewalQueue")]
        contexts = [documents[index // 5] for index in range(10)]
    else:
        history = dossier("ShipmentChronology", rows=64)
        contexts = []
        for index in range(10):
            history += (
                f"\nTool observation {index}: record {100 + index} is present, "
                "its owner was notified, and archived evidence is current."
            )
            contexts.append(history)
    return [
        f"{context}\nFor record {100 + index}, reply with just the retention days."
        for index, context in enumerate(contexts)
    ]


def chat_prefix(name="ReviewBoard"):
    system = {"role": "system", "content": "Answer using only the supplied archive. Give the number only."}
    original = {"role": "user", "content": dossier(name, rows=65)}
    return [system, original, {"role": "assistant", "content": "Ready."}]


def chat_query(index):
    return {"role": "user", "content": f"What is the retention period for record {100 + index}?"}


def chat_case(url, scenario, max_tokens):
    messages = chat_prefix()
    rows = []
    for index in range(10):
        query = chat_query(index)
        row = send_request(url, [*messages, query], max_tokens)
        rows.append(row)
        if scenario == "chat_history":
            messages.extend([query, {"role": "assistant", "content": row["text"]}])
    return rows


def parallel_case(url, scenario, max_tokens):
    prefixes = [chat_prefix()]
    if scenario == "parallel_mixed_forks":
        prefixes.append(chat_prefix("SecondReviewBoard"))
    for prefix in prefixes:
        send_request(url, [*prefix, chat_query(18)], max_tokens)
    prompts = [
        [*prefixes[index % len(prefixes)], chat_query(index)]
        for index in range(12)
    ]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(lambda prompt: send_request(url, prompt, max_tokens), prompts))
    return rows, (time.perf_counter() - started) * 1000


def run_case(url, scenario, max_tokens):
    flush(url)
    send_request(url, "Local preflight. Reply cobalt.", max_tokens=6)
    if scenario in LONG_SCENARIOS:
        rows = [send_request(url, prompt, max_tokens) for prompt in long_workload(scenario)]
    elif scenario in CHAT_SCENARIOS:
        rows = chat_case(url, scenario, max_tokens)
    elif scenario in PARALLEL_SCENARIOS:
        rows, elapsed_ms = parallel_case(url, scenario, max_tokens)
    else:
        rows = run_workload(url, scenario, max_tokens)
    if scenario not in PARALLEL_SCENARIOS:
        elapsed_ms = sum(row["total_ms"] for row in rows)
    flush(url)
    return rows, elapsed_ms


def run_process(args, mode, round_number, directory):
    port = open_port()
    url = f"http://127.0.0.1:{port}"
    budget = args.memory_mib << 20
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", str(args.model.resolve()), "--served-model-name", "default_model",
        "--host", "127.0.0.1", "--port", str(port), "--random-seed", "21",
        "--max-total-tokens", str(args.slots), "--context-length", str(args.context),
        "--enable-cache-report", "--disable-decode-cuda-graph",
        "--disable-prefill-cuda-graph",
    ]
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": os.pathsep.join(str(p.resolve()) for p in (args.source, args.deps)),
        "HF_HUB_OFFLINE": "1",
        "SGLANG_USE_MLX": "1",
        "SGLANG_MLX_CACHE_LIMIT_GB": "0.25",
        "SGLANG_MLX_USE_CUSTOM_ROPE": "0",
        "SGLANG_MLX_FUSE_SWIGLU": "0",
        "CINFER_MLX_VARIANT": mode,
        "CINFER_MLX_MEMORY_BYTES": str(budget),
        "CINFER_PROFILE_EXTEND": "1" if args.profile_extend else "0",
    })
    log_path = directory / f"{round_number:02d}-{mode}.log"
    cases, durations = {}, {}
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_ready(process, log_path)
            for scenario in args.scenarios:
                repetitions = args.parallel_repetitions if scenario in PARALLEL_SCENARIOS else 1
                cases[scenario], durations[scenario] = [], 0.0
                for _ in range(repetitions):
                    rows, duration = run_case(url, scenario, args.max_tokens)
                    cases[scenario].extend(rows)
                    durations[scenario] += duration
        except Exception as error:
            raise RuntimeError(f"Failed trial {round_number} {mode}: {log_path}") from error
        finally:
            stop(process)
    recorded = log_path.read_text(errors="replace")
    pool = POOL_PATTERN.search(recorded)
    peaks = [int(raw) for raw in re.findall(r"CINFER_MLX_PEAK bytes=(\d+)", recorded)]
    if not pool or int(pool[1]) != args.slots + 1 or not peaks:
        raise RuntimeError(f"Missing or mismatched KV/Metal telemetry: {log_path}")
    if f"CINFER_MEMORY_GUIDELINE bytes={budget}" not in recorded:
        raise RuntimeError(f"Expected memory guideline missing: {log_path}")
    slots, layers, heads, width = map(int, pool.groups()[:4])
    dtype = pool[5]
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]
    if max(peaks) > budget:
        raise RuntimeError(f"Observed Metal allocation exceeded the budget: {log_path}")
    resets = 2 * sum(
        args.parallel_repetitions if scenario in PARALLEL_SCENARIOS else 1
        for scenario in args.scenarios
    )
    if recorded.count("Cache flushed successfully!") != resets:
        raise RuntimeError(f"Required per-workload cache reset failed: {log_path}")
    if mode in ("readout", "both", "compact", "coalesced") and "CINFER_READOUT enabled=True" not in recorded:
        raise RuntimeError(f"Qwen3 readout was not active: {log_path}")
    return {
        "round": round_number, "mode": mode, "cases": cases, "log": log_path.name,
        "case_duration_ms": durations,
        "mlx_peak_allocated_bytes": max(peaks), "mlx_memory_guideline_bytes": budget,
        "grouped_prefill_widths": [
            int(width) for width in re.findall(r"CINFER_GROUP_PREFILL batch=(\d+)", recorded)
        ],
        "compact_prefills": recorded.count("CINFER_COMPACT_PREFILL"),
        "coalesced_prefill_widths": [
            int(width)
            for width in re.findall(r"CINFER_COALESCED_PREFILL batch=(\d+)", recorded)
        ],
        "pool": {"active_slots": slots - 1, "allocated_slots": slots, "dtype": dtype,
                 "bytes": slots * layers * heads * width * dtype_bytes * 2},
    }


def consistency(trials):
    baseline = {trial["round"]: trial for trial in trials if trial["mode"] == "stock"}
    differences = []
    for trial in trials:
        stock = baseline.get(trial["round"])
        if stock is None or trial is stock:
            continue
        for scenario, rows in trial["cases"].items():
            base = stock["cases"][scenario]
            assert len(rows) == len(base)
            for index, (left, right) in enumerate(zip(base, rows)):
                fields = ("text", "prompt_tokens", "output_tokens")
                if scenario not in PARALLEL_SCENARIOS:
                    fields += ("cached_tokens",)
                if any(left[k] != right[k] for k in fields):
                    differences.append({"round": trial["round"], "mode": trial["mode"],
                                        "scenario": scenario, "index": index,
                                        "fields": [k for k in fields if left[k] != right[k]]})
    return differences


def compare(trials):
    stock_by_round = {trial["round"]: trial for trial in trials if trial["mode"] == "stock"}
    matches = {}
    for trial in trials:
        if trial["mode"] == "stock" or trial["round"] not in stock_by_round:
            continue
        stock = stock_by_round[trial["round"]]
        for scenario, rows in trial["cases"].items():
            reference = stock["cases"][scenario]
            reference_ms = stock.get("case_duration_ms", {}).get(
                scenario, sum(row["total_ms"] for row in reference)
            )
            candidate_ms = trial.get("case_duration_ms", {}).get(
                scenario, sum(row["total_ms"] for row in rows)
            )
            base_first = sorted(row["first_text_ms"] for row in reference)
            candidate_first = sorted(row["first_text_ms"] for row in rows)
            p95_index = math.ceil(len(rows) * 0.95) - 1
            matches.setdefault((trial["mode"], scenario), []).append({
                "round": trial["round"], "stock_ms": reference_ms, "candidate_ms": candidate_ms,
                "reduction_pct": (1 - candidate_ms / reference_ms) * 100,
                "stock_first_text_median_ms": statistics.median(base_first),
                "candidate_first_text_median_ms": statistics.median(candidate_first),
                "stock_first_text_p95_ms": base_first[p95_index],
                "candidate_first_text_p95_ms": candidate_first[p95_index],
            })
    return [
        {"mode": mode, "scenario": scenario, "paired_trials": pairs,
         "paired_median_reduction_pct": statistics.median(row["reduction_pct"] for row in pairs),
         "wins": sum(row["candidate_ms"] < row["stock_ms"] for row in pairs),
         "first_text_median_wins": sum(row["candidate_first_text_median_ms"] <
                                       row["stock_first_text_median_ms"] for row in pairs)}
        for (mode, scenario), pairs in matches.items()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Patched SGLang python/ directory")
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--slots", type=int, default=3072)
    parser.add_argument("--context", type=int, default=1536)
    parser.add_argument("--memory-mib", type=int, default=3072)
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--profile-extend", action="store_true")
    parser.add_argument("--parallel-repetitions", type=int, default=1)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=("stock", "coalesced"))
    parser.add_argument("--scenarios", nargs="+",
                        choices=(*SCENARIOS, *LONG_SCENARIOS, *CHAT_SCENARIOS, *PARALLEL_SCENARIOS),
                        default=SCENARIOS)
    args = parser.parse_args()
    if min(args.rounds, args.slots, args.context, args.memory_mib, args.max_tokens,
           args.parallel_repetitions) < 1:
        parser.error("limits and number of rounds must be positive")
    if args.slots <= args.context:
        parser.error("KV slots must exceed the context limit")
    if len(set(args.modes)) != len(args.modes) or "stock" not in args.modes:
        parser.error("modes must be unique and include stock")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs = args.output.with_suffix(".logs")
    logs.mkdir(exist_ok=True)
    weight_file = args.model / "model.safetensors"
    with weight_file.open("rb") as handle:
        checkpoint = hashlib.file_digest(handle, "sha256").hexdigest()
    trials = []
    for round_number in range(1, args.rounds + 1):
        shift = (round_number - 1) % len(args.modes)
        modes = args.modes[shift:] + args.modes[:shift]
        for mode in modes:
            trial = run_process(args, mode, round_number, logs)
            if trials and trial["pool"] != trials[0]["pool"]:
                raise RuntimeError("KV pool differs between the paired processes")
            trials.append(trial)
            report = {
                "weight_sha256": checkpoint, "context": args.context,
                "parallel_repetitions": args.parallel_repetitions,
                "max_tokens": args.max_tokens, "trials": trials,
                "comparisons": compare(trials), "output_differences": consistency(trials),
            }
            partial = args.output.with_suffix(".json.partial")
            partial.write_text(json.dumps(report, indent=2), encoding="utf-8")
            partial.replace(args.output)
            print(f"{round_number} {mode} complete: peak {trial['mlx_peak_allocated_bytes'] >> 20} MiB",
                  flush=True)


if __name__ == "__main__":
    main()
