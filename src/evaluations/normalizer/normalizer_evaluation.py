import base64
import json
from pathlib import Path

from langsmith import Client, evaluate

from workflows.normalizer import normalize_file

from .statement_evaluators import (
    positive_transaction_total_accuracy,
    statement_kind_accuracy,
    statement_total_accuracy,
)
from .transaction_evaluators import (
    amount_accuracy,
    end_to_end_bucket_accuracy,
    report_bucket_accuracy,
    transaction_count,
    transaction_match_rate,
)


DATASET_NAME = "acetate-normalizer-rico"
DATASET_PATH = Path(__file__).parents[2] / "dataset.json"
FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"

BASE_EVALUATORS = [
    statement_kind_accuracy,
    statement_total_accuracy,
    transaction_count,
    transaction_match_rate,
    amount_accuracy,
    positive_transaction_total_accuracy,
    report_bucket_accuracy,
    end_to_end_bucket_accuracy,
]


def exact_workflow_match(outputs: dict, reference_outputs: dict) -> dict:
    """Pass only when every Normalizer evaluation has a perfect score."""

    results = [
        evaluator(outputs, reference_outputs) for evaluator in BASE_EVALUATORS
    ]

    scores = {result["key"]: result["score"] for result in results}

    return {
        "key": "exact_workflow_match",
        "score": all(score in (True, 1, 1.0) for score in scores.values()),
        "value": scores,
    }


EVALUATORS = [*BASE_EVALUATORS, exact_workflow_match]


def normalizer_target(inputs: dict, attachments: dict) -> dict:
    attachment = attachments.get("statement")
    if not attachment:
        raise ValueError("The evaluation example requires a statement attachment.")

    file_bytes = attachment["reader"].read()
    result = normalize_file(
        base64.b64encode(file_bytes).decode("ascii"),
        filename=inputs["fixture"],
    )

    return result.model_dump(mode="json")


def upload_dataset(
    client: Client | None = None,
    *,
    dataset_name: str = DATASET_NAME,
) -> str:
    """Upload the local fixture and reference output once."""

    client = client or Client()

    if client.has_dataset(dataset_name=dataset_name):
        return dataset_name

    definition = json.loads(DATASET_PATH.read_text())
    fixture = FIXTURE_DIRECTORY / definition["input"]["fixture"]

    if not fixture.is_file():
        raise FileNotFoundError(f"Evaluation fixture not found: {fixture}")

    dataset = client.create_dataset(
        dataset_name,
        description="End-to-end Acetate financial statement normalization.",
    )
    client.create_example(
        dataset_id=dataset.id,
        inputs=definition["input"],
        outputs=definition["reference_output"],
        attachments={"statement": ("application/pdf", fixture)},
        metadata={"dataset_id": definition["dataset_id"]},
    )

    return dataset_name


def run_evaluation(
    *,
    dataset_name: str = DATASET_NAME,
    experiment_prefix: str = "normalizer",
):
    """Run the complete Normalizer workflow against the LangSmith dataset."""

    client = Client()
    upload_dataset(client, dataset_name=dataset_name)

    return evaluate(
        normalizer_target,
        data=dataset_name,
        evaluators=EVALUATORS,
        experiment_prefix=experiment_prefix,
        description="Docling, structured extraction, and Jev categorization.",
        metadata={"workflow": "normalizer", "dataset": "rico-2026-06-10"},
        max_concurrency=1,
        client=client,
    )


if __name__ == "__main__":
    run_evaluation()
