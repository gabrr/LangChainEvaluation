from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tools import (
    FileToMarkdown,
    LLM,
    SingleModel,
    file_to_markdown_factory,
    llm_factory,
    single_model_factory,
)

from .models import (
    ExtractedStatement,
    NormalizedStatement,
    NormalizedTransaction,
    ReportBucket,
)


NORMALIZER_PROMPT = """
You normalize financial statements into structured data.

Infer statement.kind from the whole document:
- credit_card: invoice totals, due dates, card limits, cardholders, purchases, or installments.
- checking_account: deposits, withdrawals, transfers, Pix, account balances, or running balances.
- unknown: only when the document does not provide enough evidence for either type.

Extract every transaction exactly once. Preserve source-facing descriptions and visible
cardholder/card details. Use signed decimal strings for amounts. Extract installment counts
only from explicit evidence such as 3/12, 3 de 12, parcela 3/12, or dedicated columns.
Do not categorize transactions and do not invent missing values.
""".strip()


REPORT_BUCKET_QUESTION = {
    "type": "choice",
    "instructions": (
        "Classify this transaction into exactly one report bucket. When evidence "
        "overlaps, use this priority: excluded, income, installment, fixed, variable."
    ),
    "criteria": {
        "income": (
            "Money received, such as salary, deposits, payouts, interest, or other "
            "income-like inflows."
        ),
        "installment": (
            "A purchase, financing payment, or debt with explicit installment evidence "
            "such as 3/12, parcela 3 de 12, or structured installment fields."
        ),
        "fixed": (
            "A recurring commitment such as rent, internet, electricity, water, gas, "
            "insurance, phone plans, subscriptions, software plans, or memberships. "
            "The amount may vary. Examples include Netflix, Amazon Prime, Uber One, "
            "ChatGPT, Cursor, and Google Workspace. Repeated discretionary purchases "
            "are not fixed."
        ),
        "variable": (
            "An ordinary non-installment expense whose occurrence or amount is "
            "discretionary or irregular, such as groceries, restaurants, individual "
            "rides, pharmacy purchases, clothing, leisure, or one-time shopping."
        ),
        "excluded": (
            "A movement that should not count as income or spending, including transfers "
            "between owned accounts, refunds, chargebacks, reversals, credit-card bill "
            "payments, and balance adjustments."
        ),
    },
}


file_to_markdown: FileToMarkdown = file_to_markdown_factory("docling")
structured_llm: LLM = llm_factory(
    "langchain",
    model="openrouter:deepseek/deepseek-v4.1-flash",
    system_prompt=NORMALIZER_PROMPT,
    response_format=ExtractedStatement,
)
single_model: SingleModel = single_model_factory("jev")


class NormalizerState(TypedDict, total=False):
    file_base64: str
    filename: str
    markdown: str
    extracted: ExtractedStatement
    classified_transactions: list[NormalizedTransaction]
    result: NormalizedStatement


def normalizer_workflow() -> CompiledStateGraph:
    """Build the deterministic file normalization graph."""

    def convert_file(state: NormalizerState) -> dict:
        markdown = file_to_markdown.convert(
            state["file_base64"],
            filename=state["filename"],
        )

        return {"markdown": markdown}

    def normalize_markdown(state: NormalizerState) -> dict:
        response = structured_llm.prompt(state["markdown"])
        extracted = response.structured_output

        if isinstance(extracted, dict):
            extracted = ExtractedStatement.model_validate(extracted)

        if not isinstance(extracted, ExtractedStatement):
            raise TypeError("LLM must return an ExtractedStatement.")

        return {"extracted": extracted}

    def classify_transactions(state: NormalizerState) -> dict:
        classified_transactions = []

        for transaction in state["extracted"].transactions:
            response = single_model.evaluate(
                state={"transaction": transaction.model_dump(mode="json")},
                questions={"report_bucket": REPORT_BUCKET_QUESTION},
            )

            answer = response.get("answers", {}).get("report_bucket")

            if not isinstance(answer, dict):
                raise TypeError("Jev must return a report_bucket choice answer.")

            bucket = ReportBucket(answer.get("choice"))
            confidence = answer.get("confidence")
            probabilities = answer.get("probabilities")

            if not isinstance(confidence, (int, float)):
                raise TypeError("Jev must return classification confidence.")

            if not isinstance(probabilities, dict):
                raise TypeError("Jev must return classification probabilities.")

            classified_transactions.append(
                NormalizedTransaction(
                    **transaction.model_dump(),
                    report_bucket=bucket,
                    classification_confidence=confidence,
                    classification_probabilities=probabilities,
                )
            )

        return {"classified_transactions": classified_transactions}

    def consolidate(state: NormalizerState) -> dict:
        return {
            "result": NormalizedStatement(
                statement=state["extracted"].statement,
                transactions=state["classified_transactions"],
            )
        }

    builder = StateGraph(NormalizerState)
    builder.add_node("convert_file", convert_file)
    builder.add_node("normalize_markdown", normalize_markdown)
    builder.add_node("classify_transactions", classify_transactions)
    builder.add_node("consolidate", consolidate)
    builder.add_edge(START, "convert_file")
    builder.add_edge("convert_file", "normalize_markdown")
    builder.add_edge("normalize_markdown", "classify_transactions")
    builder.add_edge("classify_transactions", "consolidate")
    builder.add_edge("consolidate", END)

    return builder.compile()


normalizer = normalizer_workflow()


def normalize_file(file_base64: str, *, filename: str) -> NormalizedStatement:
    """Normalize one Base64-encoded financial statement."""

    state = normalizer.invoke(
        {
            "file_base64": file_base64,
            "filename": filename,
        }
    )

    result = state.get("result")

    if not isinstance(result, NormalizedStatement):
        raise TypeError("Normalizer workflow did not return a NormalizedStatement.")

    return result
