
from __future__ import annotations

import json
import re
from pathlib import Path

EVIDENCE_PII = Path("evidence/04_pii_demo_log.txt")
EVIDENCE_JSON = Path("evidence/04_json_demo_log.txt")

# ── 1. Imports ───────────────────────────────────────────────────────────────
GUARDRAILS_AVAILABLE = True
try:
    from guardrails import Guard
    from guardrails.validators import Validator, register_validator, PassResult, FailResult
    try:
        from guardrails.validator_base import OnFailAction
    except Exception:
        from guardrails.hub import OnFailAction  # type: ignore
except Exception:
    GUARDRAILS_AVAILABLE = False

    class OnFailAction:  # type: ignore
        FIX = "fix"

    def register_validator(*_args, **_kwargs):  # type: ignore
        def deco(cls):
            return cls
        return deco

    class Validator:  # type: ignore
        def __init__(self, on_fail=None):
            self.on_fail = on_fail

    class PassResult:  # type: ignore
        def __init__(self, value_override=None):
            self.value_override = value_override

    class FailResult:  # type: ignore
        def __init__(self, error_message="", fix_value=None):
            self.error_message = error_message
            self.fix_value = fix_value

    class _Outcome:
        def __init__(self, passed: bool, out: str):
            self.validation_passed = passed
            self.validated_output = out

    class Guard:  # type: ignore
        def __init__(self):
            self._validator = None

        def use(self, validator):
            self._validator = validator
            return self

        def validate(self, text: str):
            v = self._validator
            if v is None:
                return _Outcome(True, text)
            res = v.validate(text, metadata={})
            if isinstance(res, FailResult):
                return _Outcome(False, res.fix_value if res.fix_value is not None else text)
            if isinstance(res, PassResult) and getattr(res, "value_override", None) is not None:
                return _Outcome(True, res.value_override)
            return _Outcome(True, text)


# ── 2. PII Detector Validator ───────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        # US phone numbers like (555) 867-5309 or 555-123-4567, optional +1.
        "PHONE": r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            for m in re.findall(pattern, value):
                found.append((pii_type, m))
                redacted_text = redacted_text.replace(m, f"[{pii_type}_REDACTED]")

        if found:
            return FailResult(
                error_message=f"PII detected: {[t for t, _ in found]}",
                fix_value=redacted_text,
            )
        return PassResult()


# ── 3. JSON Formatter Validator ─────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    @staticmethod
    def _repair(text: str) -> str:
        t = text.strip()
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
        t = t.strip()
        t = t.replace("'", '"')
        t = re.sub(r",\s*([}\]])", r"\1", t)
        return t

    def validate(self, value: str, metadata: dict):
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2, ensure_ascii=True))
        except Exception:
            pass

        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            return PassResult(value_override=json.dumps(parsed, indent=2, ensure_ascii=True))
        except Exception as e:
            fallback = json.dumps({"error": "invalid_json", "detail": str(e), "raw": value}, ensure_ascii=True)
            return FailResult(error_message="Invalid JSON after repair", fix_value=fallback)


# ── 4. PII Guard demo ───────────────────────────────────────────────────────
def demo_pii_guard() -> str:
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]

    lines = []
    lines.append("\n" + "=" * 55)
    lines.append("  PII Detection Demo")
    lines.append("=" * 55)

    for label, text in test_cases:
        result = guard.validate(text)
        lines.append(f"\n[{label}] passed={result.validation_passed}")
        lines.append(f"  Input:  {text}")
        lines.append(f"  Output: {result.validated_output}")

    return "\n".join(lines) + "\n"


# ── 5. JSON Guard demo ──────────────────────────────────────────────────────
def demo_json_guard() -> str:
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma", '{"key": "value",}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]

    lines = []
    lines.append("\n" + "=" * 55)
    lines.append("  JSON Formatting Demo")
    lines.append("=" * 55)

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail"
        lines.append(f"\n[{label}] {status}")
        lines.append(f"  Input:  {text}")
        lines.append(f"  Output: {str(result.validated_output)}")

    return "\n".join(lines) + "\n"


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    EVIDENCE_PII.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_JSON.parent.mkdir(parents=True, exist_ok=True)

    pii_log = demo_pii_guard()
    json_log = demo_json_guard()

    print(pii_log)
    print(json_log)

    EVIDENCE_PII.write_text(pii_log, encoding="utf-8")
    EVIDENCE_JSON.write_text(json_log, encoding="utf-8")

    print(f"✅ Wrote evidence: {EVIDENCE_PII.as_posix()}")
    print(f"✅ Wrote evidence: {EVIDENCE_JSON.as_posix()}")


if __name__ == "__main__":
    main()

