from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class StatementKind(StrEnum):
    CREDIT_CARD = "credit_card"
    CHECKING_ACCOUNT = "checking_account"
    UNKNOWN = "unknown"


class ReportBucket(StrEnum):
    INCOME = "income"
    INSTALLMENT = "installment"
    FIXED = "fixed"
    VARIABLE = "variable"
    EXCLUDED = "excluded"


class StatementMetadata(BaseModel):
    kind: StatementKind
    institution_name: str | None = None
    currency: str | None = None
    statement_due_date: str | None = None
    statement_close_date: str | None = None
    statement_total: str | None = None
    page_count: int | None = None

    @field_validator("statement_total", mode="before")
    @classmethod
    def _statement_total_as_string(cls, value: object) -> str | None:
        return _decimal_as_string(value)


class ExtractedTransaction(BaseModel):
    date: str | None = None
    description: str | None = None
    amount: str | None = None
    currency: str | None = None
    cardholder: str | None = None
    card_last4: str | None = None
    payment_method: str | None = None
    merchant_name: str | None = None
    installments_current: int | None = None
    installments: int | None = None
    foreign_amount: str | None = None
    foreign_currency: str | None = None
    running_balance: str | None = None

    @field_validator("amount", "foreign_amount", "running_balance", mode="before")
    @classmethod
    def _amounts_as_strings(cls, value: object) -> str | None:
        return _decimal_as_string(value)


class ExtractedStatement(BaseModel):
    statement: StatementMetadata
    transactions: list[ExtractedTransaction] = Field(default_factory=list)


class NormalizedTransaction(ExtractedTransaction):
    report_bucket: ReportBucket
    classification_confidence: float = Field(ge=0, le=1)
    classification_probabilities: dict[ReportBucket, float]


class NormalizedStatement(BaseModel):
    statement: StatementMetadata
    transactions: list[NormalizedTransaction] = Field(default_factory=list)


def _decimal_as_string(value: object) -> str | None:
    if value is None:
        return None

    return str(value)
