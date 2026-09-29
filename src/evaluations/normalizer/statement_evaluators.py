from decimal import Decimal, InvalidOperation


def statement_kind_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    actual = outputs.get("statement", {}).get("kind")
    expected = reference_outputs.get("statement", {}).get("kind")

    return {
        "key": "statement_kind_accuracy",
        "score": actual == expected,
        "value": {"actual": actual, "expected": expected},
    }


def statement_total_accuracy(outputs: dict, reference_outputs: dict) -> dict:
    actual = _decimal(outputs.get("statement", {}).get("statement_total"))
    expected = _decimal(
        reference_outputs.get("statement", {}).get("statement_total")
    )

    return {
        "key": "statement_total_accuracy",
        "score": actual == expected,
        "value": {"actual": str(actual), "expected": str(expected)},
    }


def positive_transaction_total_accuracy(
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    actual = _positive_total(outputs)
    expected = _positive_total(reference_outputs)

    return {
        "key": "positive_transaction_total_accuracy",
        "score": actual == expected,
        "value": {"actual": str(actual), "expected": str(expected)},
    }


def _positive_total(payload: dict) -> Decimal:
    total = Decimal("0")

    for transaction in payload.get("transactions", []):
        amount = _decimal(transaction.get("amount"))

        if amount is not None and amount > 0:
            total += amount

    return total.quantize(Decimal("0.01"))


def _decimal(value: object) -> Decimal | None:
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, TypeError, ValueError):
        return None
