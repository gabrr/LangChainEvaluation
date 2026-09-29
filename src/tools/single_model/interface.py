from collections.abc import Mapping
from abc import ABC, abstractmethod
from typing import Any, Literal, Required, TypedDict

# NOUL — yes/no probability
# Instructions: "Does this batch contain a transaction?"
# Output:
# {
#     "type": "noul",
#     "noul": 0.97,
# }
#
# CHOICE — selects one option from the supplied criteria
# Criteria: fixed, variable, installment, unknown
# Output:
# {
#     "type": "choice",
#     "choice": "installment",
#     "confidence": 0.93,
#     "probabilities": {
#         "fixed": 0.02,
#         "variable": 0.04,
#         "installment": 0.93,
#         "unknown": 0.01,
#     },
# }
#
# SCORE — rates against ordered criteria from lowest to highest
# Criteria: incomplete, partial, mostly_complete, complete
# Output:
# {
#     "type": "score",
#     "score": 2.8,
#     "confidence": 0.91,
#     "legend": {
#         "0": "incomplete",
#         "1": "partial",
#         "2": "mostly_complete",
#         "3": "complete",
#     },
#     "probabilities": {
#         "0": 0.01,
#         "1": 0.03,
#         "2": 0.11,
#         "3": 0.85,
#     },
# }
#
# Instructions define the decision.
# Criteria define the possible outcomes or scoring levels.

type ModelState = str | dict[str, Any] | list[Any]


class SingleModelQuestion(TypedDict, total=False):
    type: Required[Literal["noul", "choice", "score"]]
    instructions: str | dict[str, Any] | list[Any] | None
    criteria: dict[str, Any] | list[Any] | None


type SingleModelQuestions = Mapping[str, SingleModelQuestion]
type SingleModelResult = dict[str, Any]


class SingleModel(ABC):
    """Interface for typed model decisions."""

    def evaluate(
        self,
        state: ModelState,
        questions: SingleModelQuestions,
    ) -> SingleModelResult:
        if not isinstance(state, (str, dict, list)):
            raise TypeError("state must be a string, dictionary, or list.")

        if not questions:
            raise ValueError("At least one question is required.")

        for name, question in questions.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Question names must be non-empty strings.")

            if question.get("type") not in {"noul", "choice", "score"}:
                raise ValueError(f"Unsupported question type for {name}.")

        result = self._evaluate(state, questions)

        if not isinstance(result, dict):
            raise TypeError("Provider must return a dictionary.")

        return result

    @abstractmethod
    def _evaluate(
        self,
        state: ModelState,
        questions: SingleModelQuestions,
    ) -> SingleModelResult:
        """Evaluate validated questions with the concrete model."""
        raise NotImplementedError
