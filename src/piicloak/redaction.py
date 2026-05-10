"""
File redaction helpers for PIICloak.

The secrets profile intentionally uses pattern recognizers only so agent-memory
transcript redaction can run without loading a spaCy model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from presidio_analyzer import RecognizerResult

from .recognizers import create_api_key_recognizer

SUPPORTED_PROFILES = ["secrets"]
MAX_JSON_DEPTH = 100

CONTEXTUAL_SECRET_VALUE_PATTERNS = [
    re.compile(
        r"(?i)(?:clickup|click_up)(?:[\s_-]+api)?[\s_-]+(?:key|token)\s*[=:]\s*"
        r"['\"]?(?P<secret>pk_[a-zA-Z0-9_-]{20,})['\"]?"
    ),
    re.compile(
        r"(?i)(?:cloudflare|cf)(?:[\s_-]+api)?[\s_-]+token\s*[=:]\s*"
        r"['\"]?(?P<secret>[a-zA-Z0-9_-]{20,})['\"]?"
    ),
    re.compile(
        r"(?i)(?:aws[_-]?secret(?:[_-]?access)?[_-]?key)\s*[=:]\s*"
        r"['\"]?(?P<secret>[a-zA-Z0-9/+=]{40})['\"]?"
    ),
    re.compile(
        r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[=:]\s*"
        r"['\"]?(?P<secret>[a-zA-Z0-9_\-]{20,})['\"]?"
    ),
    re.compile(r"(?i)bearer\s+(?P<secret>[a-zA-Z0-9_\-\.]{20,})"),
]


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
            previous_length = previous.end - previous.start
            result_length = result.end - result.start
            if result.end > previous.end:
                previous.end = result.end
            if result_length > previous_length:
                previous.entity_type = result.entity_type
                previous.score = result.score
            continue
        merged.append(result)
    return merged


@lru_cache(maxsize=1)
def get_secrets_recognizer():
    """Return the cached pattern recognizer used by the secrets profile."""
    return create_api_key_recognizer()


def analyze_secrets(text: str) -> list[RecognizerResult]:
    """Analyze text for technical secret-shaped values."""
    recognizer = get_secrets_recognizer()
    results = recognizer.analyze(text, ["API_KEY"])
    return merge_results(adjust_contextual_secret_spans(text, results))


def adjust_contextual_secret_spans(
    text: str, results: list[RecognizerResult]
) -> list[RecognizerResult]:
    """Keep useful labels while redacting only contextual secret values."""
    for result in results:
        matched_text = text[result.start : result.end]
        for pattern in CONTEXTUAL_SECRET_VALUE_PATTERNS:
            match = pattern.fullmatch(matched_text)
            if match:
                result.start += match.start("secret")
                result.end = result.start + len(match.group("secret"))
                break
    return results


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


def redact_value(value: Any, counts: Counter[str], depth: int = 0) -> Any:
    """Redact strings recursively while preserving JSON structure."""
    if depth > MAX_JSON_DEPTH:
        raise ValueError("JSON nesting is too deep to redact safely")
    if isinstance(value, str):
        return redact_text(value, counts)
    if isinstance(value, list):
        return [redact_value(item, counts, depth + 1) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item, counts, depth + 1) for key, item in value.items()}
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
        "input": input_path.name,
        "output": None if dry_run else ("-" if output == "-" else Path(output).name),
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
