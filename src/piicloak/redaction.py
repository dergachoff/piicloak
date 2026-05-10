"""
File redaction helpers for PIICloak.

The secrets profile intentionally uses pattern recognizers only so agent-memory
transcript redaction can run without loading a spaCy model.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from presidio_analyzer import RecognizerResult

from .recognizers import create_api_key_recognizer

SUPPORTED_PROFILES = ["secrets"]


def placeholder(entity_type: str) -> str:
    """Return replacement placeholder for an entity type."""
    return f"<{entity_type}>"


def merge_results(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Merge overlapping recognizer results, preferring longer spans."""
    if not results:
        return []

    sorted_results = sorted(results, key=lambda item: (item.start, -(item.end - item.start)))
    merged: list[RecognizerResult] = []
    for result in sorted_results:
        if merged and result.start < merged[-1].end:
            previous = merged[-1]
            if result.end - result.start > previous.end - previous.start:
                previous.end = max(previous.end, result.end)
                previous.entity_type = result.entity_type
                previous.score = result.score
            continue
        merged.append(result)
    return merged


def analyze_secrets(text: str) -> list[RecognizerResult]:
    """Analyze text for technical secret-shaped values."""
    recognizer = create_api_key_recognizer()
    return merge_results(recognizer.analyze(text, ["API_KEY"]))


def redact_text(text: str, counts: Counter[str]) -> str:
    """Redact secret-shaped values from a text string."""
    results = analyze_secrets(text)
    if not results:
        return text

    pieces: list[str] = []
    cursor = 0
    for result in results:
        pieces.append(text[cursor : result.start])
        pieces.append(placeholder(result.entity_type))
        counts[result.entity_type] += 1
        cursor = result.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def redact_value(value: Any, counts: Counter[str]) -> Any:
    """Redact strings recursively while preserving JSON structure."""
    if isinstance(value, str):
        return redact_text(value, counts)
    if isinstance(value, list):
        return [redact_value(item, counts) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, counts) for key, item in value.items()}
    return value


def redact_jsonl(input_path: Path, counts: Counter[str]) -> str:
    """Redact a JSONL file line by line."""
    lines: list[str] = []
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                lines.append(line)
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                lines.append(redact_text(line, counts))
                continue
            lines.append(json.dumps(redact_value(payload, counts), ensure_ascii=False) + "\n")
    return "".join(lines)


def redact_json(input_path: Path, counts: Counter[str]) -> str:
    """Redact a JSON file while preserving JSON structure."""
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    return json.dumps(redact_value(payload, counts), ensure_ascii=False, indent=2) + "\n"


def redact_file(input_path: Path) -> tuple[str, Counter[str]]:
    """Redact a supported file and return redacted content plus counts."""
    counts: Counter[str] = Counter()
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        redacted = redact_jsonl(input_path, counts)
    elif suffix == ".json":
        redacted = redact_json(input_path, counts)
    else:
        redacted = redact_text(input_path.read_text(encoding="utf-8", errors="replace"), counts)
    return redacted, counts


def build_summary(
    input_path: Path, output: str, counts: Counter[str], dry_run: bool
) -> dict[str, Any]:
    """Build a safe summary without raw matched secret values."""
    return {
        "ok": True,
        "profile": "secrets",
        "dry_run": dry_run,
        "input": str(input_path),
        "output": None if dry_run else output,
        "redactions": dict(counts),
        "total_redactions": sum(counts.values()),
    }


def write_output(output: str, content: str, summary: dict[str, Any]) -> None:
    """Write redacted content and emit a safe summary."""
    if output == "-":
        sys.stdout.write(content)
        print(json.dumps(summary, sort_keys=True), file=sys.stderr)
        return

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def redact_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for file redaction."""
    parser = argparse.ArgumentParser(description="Redact PIICloak-supported files")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES, default="secrets")
    parser.add_argument("--input", required=True, help="Input .jsonl, .json, .txt, or .md file")
    parser.add_argument("--output", default="-", help="Output file path, or '-' for stdout")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report counts without writing redacted content"
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    redacted, counts = redact_file(input_path)
    summary = build_summary(input_path, args.output, counts, args.dry_run)
    if args.dry_run:
        print(json.dumps(summary, sort_keys=True))
        return 0

    write_output(args.output, redacted, summary)
    return 0
