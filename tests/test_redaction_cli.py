"""Tests for the file redaction CLI."""

import json
import subprocess
import sys
from collections import Counter

import pytest

import piicloak.__main__ as cli
import piicloak.redaction
from piicloak.redaction import build_summary, redact_file, redact_main, redact_text


def run_cli(*args):
    """Run the PIICloak module CLI."""
    return subprocess.run(
        [sys.executable, "-m", "piicloak", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def token(*parts):
    """Build fake token-shaped test values without storing full fixtures."""
    return "".join(parts)


def test_redact_jsonl_file_preserves_agent_memory_context(tmp_path):
    """Test JSONL transcript redaction preserves useful non-secret identifiers."""
    source = tmp_path / "session.jsonl"
    output = tmp_path / "session.redacted.jsonl"
    openrouter_token = token("sk-or-", "v1-abcdefghijklmnopqrstuvwxyz123456")
    slack_token = token("xox", "b-123456789012-123456789012-abcdefghijklmnopqrstuvwx")
    lines = [
        {
            "type": "event_msg",
            "payload": {
                "type": "user_message",
                "message": (
                    "Commit 1eeb16dd9f0c2a3b4d5e6f708192a3b4c5d6e7f8 "
                    "session 019df702-35e9-7af3-aef3-d3baf62835b6 "
                    f"OpenRouter {openrouter_token}"
                ),
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "assistant_message",
                "message": f"Slack {slack_token}",
            },
        },
    ]
    source.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    result = run_cli(
        "redact",
        "--profile",
        "secrets",
        "--input",
        str(source),
        "--output",
        str(output),
    )

    summary = json.loads(result.stdout)
    redacted = output.read_text(encoding="utf-8")
    assert summary["redactions"] == {"API_KEY": 2}
    assert summary["total_redactions"] == 2
    assert "sk-or-v1-" not in redacted
    assert "xoxb-" not in redacted
    assert "<API_KEY>" in redacted
    assert "1eeb16dd9f0c2a3b4d5e6f708192a3b4c5d6e7f8" in redacted
    assert "019df702-35e9-7af3-aef3-d3baf62835b6" in redacted


def test_redact_json_and_plain_text(tmp_path):
    """Test JSON and text inputs are supported."""
    json_source = tmp_path / "payload.json"
    text_source = tmp_path / "notes.txt"
    json_output = tmp_path / "payload.redacted.json"
    anthropic_token = token("sk-ant-", "api03-abcdefghijklmnopqrstuvwxyz123456")
    telegram_token = token("1234567890", ":ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi")

    json_source.write_text(
        json.dumps({"token": f"Anthropic {anthropic_token}"}),
        encoding="utf-8",
    )
    text_source.write_text(
        f"Telegram {telegram_token}",
        encoding="utf-8",
    )

    json_result = run_cli(
        "redact",
        "--input",
        str(json_source),
        "--output",
        str(json_output),
    )
    text_result = run_cli("redact", "--input", str(text_source), "--output", "-")

    assert json.loads(json_result.stdout)["total_redactions"] == 1
    assert "sk-ant-" not in json_output.read_text(encoding="utf-8")
    assert "<API_KEY>" in json_output.read_text(encoding="utf-8")

    text_summary = json.loads(text_result.stderr)
    assert text_summary["total_redactions"] == 1
    assert "1234567890:" not in text_result.stdout
    assert "<API_KEY>" in text_result.stdout


def test_redact_jsonl_falls_back_to_text_for_malformed_lines(tmp_path):
    """Test malformed JSONL lines are still redacted as text."""
    source = tmp_path / "mixed.jsonl"
    output = tmp_path / "mixed.redacted.jsonl"
    github_token = token("gh", "p_abcdefghijklmnopqrstuvwxyz123456")
    gitlab_token = token("gl", "pat-abcdefghijklmnopqrstuvwx")
    source.write_text(
        f'\n{{"message": "GitHub {github_token}"}}\n' f"broken token {gitlab_token}\n",
        encoding="utf-8",
    )

    result = run_cli("redact", "--input", str(source), "--output", str(output))

    summary = json.loads(result.stdout)
    redacted = output.read_text(encoding="utf-8")
    assert summary["total_redactions"] == 2
    assert redacted.startswith("\n")
    assert "ghp_" not in redacted
    assert "glpat-" not in redacted
    assert redacted.count("<API_KEY>") == 2


def test_redact_dry_run_does_not_write_output(tmp_path):
    """Test dry-run reports counts without creating output."""
    source = tmp_path / "session.jsonl"
    output = tmp_path / "session.redacted.jsonl"
    gitlab_token = token("gl", "pat-abcdefghijklmnopqrstuvwx")
    source.write_text(
        json.dumps({"message": f"GitLab {gitlab_token}"}) + "\n",
        encoding="utf-8",
    )

    result = run_cli(
        "redact",
        "--profile",
        "secrets",
        "--input",
        str(source),
        "--output",
        str(output),
        "--dry-run",
    )

    summary = json.loads(result.stdout)
    assert summary["dry_run"] is True
    assert summary["output"] is None
    assert summary["total_redactions"] == 1
    assert not output.exists()


def test_redaction_helpers_count_without_raw_secret_values(tmp_path):
    """Test redaction helpers return safe counts and no raw matched values."""
    source = tmp_path / "notes.md"
    huggingface_token = token("hf", "_abcdefghijklmnopqrstuvwxyz1234567890")
    source.write_text(
        f"Commit 1eeb16dd keeps context, token {huggingface_token}",
        encoding="utf-8",
    )

    redacted, counts = redact_file(source)
    summary = build_summary(source, "-", counts, dry_run=True)

    assert counts == {"API_KEY": 1}
    assert summary["redactions"] == {"API_KEY": 1}
    assert "hf_" not in redacted
    assert "1eeb16dd" in redacted
    assert "abcdefghijklmnopqrstuvwxyz" not in json.dumps(summary)


def test_redact_text_keeps_plain_text_without_secrets():
    """Test text without secrets is unchanged."""
    text = "Keep PR #2, commit 1eeb16dd, and domain example.com."
    counts = Counter()

    redacted = redact_text(text, counts)

    assert redacted == text
    assert counts == {}


def test_redact_main_writes_stdout_and_safe_summary_to_stderr(tmp_path, capsys):
    """Test in-process CLI writes redacted stdout and safe summary stderr."""
    source = tmp_path / "notes.txt"
    linear_token = token("lin", "_api_abcdefghijklmnopqrstuvwxyz123456")
    source.write_text(
        f"Linear {linear_token}",
        encoding="utf-8",
    )

    exit_code = redact_main(["--input", str(source), "--output", "-"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "lin_api_" not in captured.out
    assert "<API_KEY>" in captured.out
    summary = json.loads(captured.err)
    assert summary["total_redactions"] == 1
    assert "abcdefghijklmnopqrstuvwxyz" not in captured.err


def test_module_main_dispatches_redact(monkeypatch):
    """Test console entrypoint dispatches to the redact subcommand."""
    calls = []
    monkeypatch.setattr(sys, "argv", ["piicloak", "redact", "--dry-run"])
    monkeypatch.setattr(
        piicloak.redaction,
        "redact_main",
        lambda argv: calls.append(argv) or 0,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert calls == [["--dry-run"]]
