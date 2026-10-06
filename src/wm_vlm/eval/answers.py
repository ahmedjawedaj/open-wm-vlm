"""Strict parsing and scoring of multiple-choice answers.

Model outputs are free text. This module turns one output into either a valid
option index or an explicit failure status. It never guesses: a letter that only
appears inside an explanation is not an answer.

Accepted forms
--------------
Matching is case-insensitive and ignores surrounding whitespace, Markdown
emphasis (``*``, ``_``, backticks) and quotes. The output is split into lines.
A line is an *answer declaration* when it is one of:

* a marked declaration, an answer marker followed only by a label:
  ``Answer: B``, ``Final answer: (B)``, ``The answer is B.``,
  ``Correct answer: option b``, ``Answer B``
* a bare label on its own line: ``B``, ``b``, ``(B)``, ``[B]``, ``B.``, ``B)``,
  ``Option B``

Lines without an answer marker or a bare declaration are treated as explanation
and ignored, even when they contain option letters. Reasoning followed by a final
declaration line is therefore valid.

Marked declarations take precedence, including malformed declarations. A broken
``Answer:`` line invalidates the output rather than falling back to an earlier
label. Bare labels are used only when there is no marked declaration. Option
headings such as ``Option A:`` are not declarations.

Failure statuses
----------------
``MISSING``       the output is ``None`` or contains only whitespace
``AMBIGUOUS``     declarations name more than one label, for example
                  ``Answer: A or B``, ``Answer: A`` followed by ``Answer: B``,
                  or several bare labels on separate lines
``OUT_OF_RANGE``  the single declared label is a letter beyond the option count
``MALFORMED``     no declaration was found, or a marked declaration is malformed

Repeating the same single label in several declarations is still valid.

Scoring
-------
:func:`score_outputs` counts every requested item in the denominator. Any
non-valid output is incorrect and is never dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

DEFAULT_NUM_OPTIONS = 4
MAX_OPTIONS = 26

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DECORATION = " \t*_`\"'"
_LABEL_BODY = r"(?a:\([A-Za-z]\)|\[[A-Za-z]\]|[A-Za-z]\)?)"
_LABEL_ATOM = rf"(?:(?:options?|choices?|letters?)\s+)?{_LABEL_BODY}"
_DECLARATION = re.compile(
    rf"{_LABEL_ATOM}(?:(?:[\s,/&|]+|\s+(?:and|or)\s+){_LABEL_ATOM})*",
    re.IGNORECASE,
)
_LABELS_IN_DECLARATION = re.compile(r"\b[A-Za-z]\b")
_MARKER = re.compile(
    r"(?:the\s+)?(?:(?:final|correct)\s+)?answer\b\s*(?:is\b\s*)?[:=]?\s*(?P<rest>.*)",
    re.IGNORECASE,
)


class ParseStatus(str, Enum):
    """Outcome of parsing one model output."""

    VALID = "valid"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    OUT_OF_RANGE = "out_of_range"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class ParsedAnswer:
    """Parse result. ``raw`` is the unmodified model output."""

    raw: str | None
    status: ParseStatus
    index: int | None = None

    @property
    def is_valid(self) -> bool:
        return self.status is ParseStatus.VALID

    @property
    def label(self) -> str | None:
        """Option letter, or ``None`` when the output is not valid."""
        return None if self.index is None else _LABELS[self.index]


def _declared_labels(line: str) -> tuple[list[str], bool] | None:
    """Return labels and marker presence, retaining malformed marked lines.

    An empty label list with a marker denotes malformed syntax. Only validated
    declaration bodies are searched for labels, so prose is never extracted.
    """
    text = line.strip(_DECORATION)
    marker = _MARKER.fullmatch(text)
    if marker is not None:
        text = marker.group("rest").strip(_DECORATION)
    text = text.rstrip(".!").strip(_DECORATION)
    if _DECLARATION.fullmatch(text) is None:
        return ([], True) if marker is not None else None
    labels = [label.upper() for label in _LABELS_IN_DECLARATION.findall(text)]
    return labels, marker is not None


def parse_answer(raw: str | None, num_options: int = DEFAULT_NUM_OPTIONS) -> ParsedAnswer:
    """Parse one model output into an option index or an explicit failure status.

    Labels ``A``, ``B``, ... map to indices ``0``, ``1``, ... and only the first
    ``num_options`` letters are valid.
    """
    if not isinstance(num_options, int) or isinstance(num_options, bool):
        raise TypeError("num_options must be an int")
    if not 2 <= num_options <= MAX_OPTIONS:
        raise ValueError(f"num_options must be between 2 and {MAX_OPTIONS}, got {num_options}")
    if raw is not None and not isinstance(raw, str):
        raise TypeError(f"model output must be str or None, got {type(raw).__name__}")
    if raw is None or not raw.strip():
        return ParsedAnswer(raw, ParseStatus.MISSING)

    found = [d for line in raw.splitlines() if (d := _declared_labels(line)) is not None]
    marked = [labels for labels, is_marked in found if is_marked]
    declarations = marked or [labels for labels, _ in found]
    if not declarations:
        return ParsedAnswer(raw, ParseStatus.MALFORMED)
    if any(len(labels) > 1 for labels in declarations):
        return ParsedAnswer(raw, ParseStatus.AMBIGUOUS)
    distinct = {labels[0] for labels in declarations if labels}
    if len(distinct) > 1:
        return ParsedAnswer(raw, ParseStatus.AMBIGUOUS)
    if any(not labels for labels in declarations):
        return ParsedAnswer(raw, ParseStatus.MALFORMED)
    index = _LABELS.index(next(iter(distinct)))
    if index >= num_options:
        return ParsedAnswer(raw, ParseStatus.OUT_OF_RANGE)
    return ParsedAnswer(raw, ParseStatus.VALID, index)


def is_correct(parsed: ParsedAnswer, answer: int) -> bool:
    """True only for a valid parse that selects the correct option index."""
    return parsed.is_valid and parsed.index == answer


@dataclass(frozen=True)
class ScoreReport:
    """Accuracy over all requested items, with invalid outputs counted as wrong."""

    total: int
    correct: int
    status_counts: dict[str, int]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def invalid(self) -> int:
        return self.total - self.status_counts.get(ParseStatus.VALID.value, 0)


def score_outputs(
    outputs: Sequence[str | None],
    answers: Sequence[int],
    num_options: int = DEFAULT_NUM_OPTIONS,
) -> ScoreReport:
    """Score model outputs against correct option indices.

    Every item stays in the denominator. Missing, ambiguous, out-of-range and
    malformed outputs are incorrect.
    """
    if len(outputs) != len(answers):
        raise ValueError(f"got {len(outputs)} outputs for {len(answers)} answers")
    if not answers:
        raise ValueError("cannot score an empty set of items")
    for answer in answers:
        if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer < num_options:
            raise ValueError(f"answer index {answer!r} is not in range(0, {num_options})")
    parsed = [parse_answer(raw, num_options) for raw in outputs]
    correct = sum(is_correct(p, a) for p, a in zip(parsed, answers, strict=True))
    counts = Counter(p.status.value for p in parsed)
    return ScoreReport(
        total=len(answers),
        correct=correct,
        status_counts={status.value: counts.get(status.value, 0) for status in ParseStatus},
    )
