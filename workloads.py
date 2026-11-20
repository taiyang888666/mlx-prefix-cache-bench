"""HTTP workloads carried forward from the prior prototype experiment."""

import json
import socket
import time
from urllib.request import Request, urlopen

SCENARIOS = ("branches", "mixed", "linear", "cold")


def dossier(name, rows=28):
    clauses = []
    for index in range(rows):
        clauses.append(
            f"Record {100 + index}. Region {index % 7}; retention {17 + index % 31} days. "
            f"Owner: department {index % 9}. Approval requires record of evidence "
            f"and reference code {2040 + index * 7}.\n"
        )
    return f"Archive {name}. Office workflow reference.\n" + "".join(clauses)


def workload(scenario):
    alpha = dossier("Albatross")
    bravo = dossier("Birch")
    request = "\nQuestion {index}: Give the retention days for record {record}. Reply with the number only."

    def branch(document, index):
        return document + request.format(index=index, record=100 + index % 20)

    if scenario == "branches":
        return [branch(alpha if index < 6 else bravo, index) for index in range(12)]
    if scenario == "mixed":
        # The hot document becomes established before independent long contexts arrive.
        return [
            branch(dossier(f"Unrelated{index}"), index) if index in (3, 5, 7, 9, 11)
            else branch(alpha, index)
            for index in range(13)
        ]
    if scenario == "linear":
        growing = dossier("Linear", rows=11)
        turns = []
        for index in range(10):
            growing += f"\nTool observation step {index}. Received record ID {100 + index}; verified."
            turns.append(branch(growing, index))
        return turns
    return [branch(dossier(f"Cold{index}"), index) for index in range(8)]


def send_request(base_url, prompt, max_tokens):
    payload = {
        "model": "default_model",
        "messages": prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    usage = None
    first = None
    started = time.perf_counter()
    with urlopen(request, timeout=90) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice["delta"]
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if text:
                    if first is None:
                        first = time.perf_counter() - started
                    chunks.append(text)
    elapsed = time.perf_counter() - started
    if first is None or usage is None:
        raise RuntimeError("inference stream has no text or no final usage")
    if not 0 <= usage["prompt_tokens_details"]["cached_tokens"] < usage["prompt_tokens"]:
        raise RuntimeError("server reported invalid cached token count")
    return {
        "first_text_ms": first * 1000,
        "total_ms": elapsed * 1000,
        "prompt_tokens": usage["prompt_tokens"],
        "cached_tokens": usage["prompt_tokens_details"]["cached_tokens"],
        "output_tokens": usage["completion_tokens"],
        "text": "".join(chunks),
    }


def run_workload(base_url, scenario, max_tokens):
    results = []
    for index, prompt in enumerate(workload(scenario)):
        row = send_request(base_url, prompt, max_tokens)
        row["expected_answer"] = str(17 + index % 20)
        row["answer_matches_reference"] = row["text"].strip() == row["expected_answer"]
        results.append(row)
    return results


def open_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]
