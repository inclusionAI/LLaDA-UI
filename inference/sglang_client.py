#!/usr/bin/env python3
"""Minimal OpenAI-compatible SGLang client for LLaDA-UI.

The three bundled JSON files preserve the actual mobile, desktop, and web
evaluation prompts, histories, screenshots, and decoding parameters.  This
client uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXAMPLES = {
    "mobile": HERE / "examples" / "mobile.json",
    "desktop": HERE / "examples" / "desktop.json",
    "web": HERE / "examples" / "web.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--example", choices=sorted(EXAMPLES))
    source.add_argument("--request", type=Path, help="OpenAI-compatible JSON payload")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY", "empty"))
    parser.add_argument("--model", default=os.environ.get("SGLANG_MODEL", "LLaDA-UI"))
    return parser.parse_args()


def print_response(response, streaming: bool) -> None:
    if not streaming:
        print(response.read().decode("utf-8"))
        return
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            line = line[6:]
        try:
            event = json.loads(line)
            # A streaming chunk may carry an empty "choices": [] (e.g. the final
            # usage-only chunk); guard against IndexError on [0].
            choices = event.get("choices") or [{}]
            delta = choices[0].get("delta", {}).get("content")
            if delta:
                print(delta, end="", flush=True)
        except json.JSONDecodeError:
            print(line)
    print()


def validate_packaged_example(payload: dict) -> None:
    """Verify the released current-image-only multi-turn message contract."""
    messages = payload.get("messages", [])
    roles = [message.get("role") for message in messages]
    expected = ["system", "user", "assistant", "user"]
    if roles != expected:
        raise ValueError(f"expected packaged-example roles {expected}, got {roles}")
    if messages[1].get("content") != "":
        raise ValueError("the historical user message must have empty content")
    previous = messages[2].get("content", "")
    if not isinstance(previous, str) or "<think>" not in previous or "<action>" not in previous:
        raise ValueError("the previous assistant message must retain the raw tagged response")
    current = messages[3].get("content")
    if not isinstance(current, list) or [part.get("type") for part in current] != ["text", "image_url"]:
        raise ValueError("the current user turn must contain text first and one image last")
    image_count = sum(
        part.get("type") == "image_url"
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )
    if image_count != 1:
        raise ValueError(f"expected exactly one current image, got {image_count}")


def main() -> None:
    args = parse_args()
    path = EXAMPLES[args.example] if args.example else args.request
    payload = json.loads(path.read_text(encoding="utf-8"))
    if args.example:
        validate_packaged_example(payload)
    payload["model"] = args.model

    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            print_response(response, bool(payload.get("stream")))
    except urllib.error.HTTPError as error:
        sys.stderr.write(error.read().decode("utf-8", errors="replace") + "\n")
        raise SystemExit(error.code) from error


if __name__ == "__main__":
    main()
