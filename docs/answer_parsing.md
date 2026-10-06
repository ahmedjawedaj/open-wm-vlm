# Answer parsing

`wm_vlm.eval` turns free-text model output into a valid option index or an explicit failure status. It is CPU-only, imports no model provider and never guesses a letter from prose. It is infrastructure for future inference adapters. This repository does not run a model and reports no benchmark scores.

## Example

```python
from wm_vlm.eval import ParseStatus, parse_answer, score_outputs

parsed = parse_answer("The shape turns about X.\nFinal answer: (c)")
assert parsed.status is ParseStatus.VALID
assert (parsed.index, parsed.label) == (2, "C")

for raw in ["Answer: A or B", "Answer: E", "I think it is probably B", None]:
    result = parse_answer(raw)
    assert not result.is_valid and result.index is None
    assert result.raw == raw  # the model output is always preserved

outputs = ["B", "Answer: C", "A or B", None]
report = score_outputs(outputs, answers=[1, 2, 0, 3])
assert (report.total, report.correct, report.invalid) == (4, 2, 2)
assert report.accuracy == 0.5
assert report.status_counts["ambiguous"] == 1 and report.status_counts["missing"] == 1
```

Use `sample["answer"]` from a verified dataset record as the correct index. Never read the answer from the predictor's own output.

## Accepted forms

Matching ignores case, surrounding whitespace, Markdown emphasis (`*`, `_`, backticks) and quotes. The output is split into lines and each line is checked on its own.

| Kind | Examples | Result |
|---|---|---|
| Marked declaration | `Answer: B`, `answer b`, `Final answer: (B)`, `The answer is B.`, `Correct answer: option b` | valid |
| Bare label on its own line | `B`, `b`, `(B)`, `[B]`, `B.`, `B)`, `Option B` | valid |
| Any other line | `Option A is wrong.`, `A because it rotates` | ignored as explanation |

Reasoning followed by a final declaration line is valid. Letters inside explanations are never extracted.

Marked declarations take precedence. Bare labels count only when the output has no marked declaration, so headings such as `Option A:` in an explanation do not conflict with a final `Answer: C`.

Repeating the same single label in several declarations is valid.

## Failure statuses

| Status | Meaning | Examples |
|---|---|---|
| `missing` | `None` or whitespace only | `None`, `""`, `"\n"` |
| `ambiguous` | declarations name more than one label | `Answer: A or B`, `Answer: A, B`, `Answer: A` then `Answer: B`, `A` then `B` on separate lines |
| `out_of_range` | the single declared label is a letter beyond the option count | `Answer: E` with four options |
| `malformed` | no declaration was found | `I think it is probably B`, `A because it rotates`, `Answer: 2`, `Answer: AB` |

A conflict takes priority over a range error. `Answer: A` followed by `Answer: E` is `ambiguous`.

`parse_answer(raw, num_options=4)` accepts two to 26 options and maps `A`, `B`, ... to indices `0`, `1`, ... Anything other than `str` or `None` raises `TypeError` because it indicates a bug in the caller, not a model failure.

## Scoring

`score_outputs(outputs, answers, num_options=4)` counts every item in the denominator. Missing, ambiguous, out-of-range and malformed outputs are incorrect and are never dropped. The report holds `total`, `correct`, `accuracy`, `invalid` (outputs that did not parse to a valid option) and a count for every status. It raises `ValueError` for mismatched lengths, empty input or an answer index outside the option range.

## Limits

The parser is deliberately strict. A model that writes `The answer is B because the bar is vertical` is `malformed`, so prompts should ask for a final line such as `Answer: B`. Parsing success says nothing about whether the model reasoned correctly. Report the status counts next to any accuracy figure so format failures stay visible.
