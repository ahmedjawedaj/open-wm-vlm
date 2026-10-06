"""Strict multiple-choice parsing and scoring (issue #1)."""

import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from wm_vlm.data.config import DatasetConfig
from wm_vlm.data.rng import DetRng
from wm_vlm.data.tetris import OPTION_LABELS, generate_sample
from wm_vlm.eval import ParsedAnswer, ParseStatus, is_correct, parse_answer, score_outputs
from wm_vlm.eval import answers as answers_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("raw", "index"),
    [
        ("A", 0),
        ("b", 1),
        ("  C  ", 2),
        ("\n\tD\n", 3),
        ("(B)", 1),
        ("[c]", 2),
        ("D.", 3),
        ("B)", 1),
        ("**A**", 0),
        ("`C`", 2),
        ('"d"', 3),
        ("Option B", 1),
        ("Answer: A", 0),
        ("answer: b", 1),
        ("ANSWER:C", 2),
        ("Answer D", 3),
        ("Answer: (B)", 1),
        ("Answer: option c", 2),
        ("Final answer: D", 3),
        ("Final Answer: [A]", 0),
        ("Correct answer: B", 1),
        ("The answer is C", 2),
        ("The answer is: B.", 1),
        ("The final answer is D!", 3),
        ("**Answer: B**", 1),
        ("  Answer :  a  ", 0),
        ("Answer: A\r\n", 0),
    ],
)
def test_accepted_forms(raw, index):
    parsed = parse_answer(raw)
    assert parsed.status is ParseStatus.VALID
    assert parsed.index == index
    assert parsed.label == "ABCD"[index]
    assert parsed.is_valid


def test_final_declaration_after_reasoning_is_accepted():
    raw = (
        "Rotating the query 90 degrees about X gives a vertical bar.\n"
        "Option A keeps the bar flat, option C mirrors it and option B has the wrong count.\n"
        "Answer: D"
    )
    assert parse_answer(raw).index == 3


def test_explanation_mentioning_letters_after_the_answer_is_ignored():
    parsed = parse_answer("Answer: B\nExplanation: A and C are rotations about the wrong axis.")
    assert parsed.status is ParseStatus.VALID
    assert parsed.index == 1


def test_repeating_the_same_label_is_still_valid():
    assert parse_answer("Answer: C\nFinal answer: C").index == 2


def test_option_headings_do_not_conflict_with_a_marked_answer():
    raw = "Option A:\nwrong axis\nOption B:\nwrong count\nAnswer: C"
    assert parse_answer(raw).index == 2


@pytest.mark.parametrize(
    "raw",
    [
        "B\nAnswer: 2",
        "Answer: B\nFinal answer:",
        "Answer: B\nFinal answer: C because it rotates",
        "Final answer: C because it rotates\nAnswer: B",
        "Answer: B\nAnswer: A or",
        "Answer: B\nAnswer: (C]",
        "Answer:\nB",
    ],
)
def test_malformed_marked_answers_never_fall_back_to_valid_labels(raw):
    parsed = parse_answer(raw)
    assert parsed.status is ParseStatus.MALFORMED
    assert parsed.index is None
    assert parsed.raw == raw
    report = score_outputs([raw], [1])
    assert (report.total, report.correct, report.invalid) == (1, 0, 1)


@pytest.mark.parametrize(
    "raw",
    [
        "Answer: A or",
        "Answer: or B",
        "Answer: A and",
        "Answer: A/",
        "Answer: A,",
        "Answer: option",
        "Answer: (B]",
        "Answer: [B)",
        "Answer: (B",
        "Answer: B]",
        "Answer: A K",
        "Answer: A İ",
        "Answer: A ı",
        "Answer: A ſ",
        "Option B:",
        "B:",
    ],
)
def test_incomplete_lists_and_malformed_label_syntax_are_invalid(raw):
    parsed = parse_answer(raw)
    assert parsed.status is ParseStatus.MALFORMED
    assert parsed.index is None


@pytest.mark.parametrize("raw", [None, "", " ", "\n\n", "\t \r\n"])
def test_missing_output(raw):
    parsed = parse_answer(raw)
    assert parsed.status is ParseStatus.MISSING
    assert parsed.index is None
    assert parsed.label is None


@pytest.mark.parametrize(
    "raw",
    [
        "Answer: A or B",
        "Answer: A, B",
        "Answer: A and C",
        "Answer: A/B",
        "Answer: A B",
        "A or B",
        "A, B, C, D",
        "Answer: A\nAnswer: B",
        "Answer: B\nFinal answer: C",
        "Answer: A\nAnswer: E",
        "A\nB",
        "A\nB\nC\nD",
        "Answer: A, A",
    ],
)
def test_conflicting_labels_are_ambiguous(raw):
    parsed = parse_answer(raw)
    assert parsed.status is ParseStatus.AMBIGUOUS
    assert parsed.index is None


@pytest.mark.parametrize(
    "raw",
    ["E", "Answer: E", "answer: (f)", "Final answer: Z", "The answer is H."],
)
def test_labels_beyond_four_options_are_out_of_range(raw):
    parsed = parse_answer(raw)
    assert parsed.status is ParseStatus.OUT_OF_RANGE
    assert parsed.index is None


def test_option_count_controls_the_valid_range():
    assert parse_answer("C", num_options=2).status is ParseStatus.OUT_OF_RANGE
    assert parse_answer("B", num_options=2).index == 1
    assert parse_answer("Answer: F", num_options=6).index == 5
    assert parse_answer("Answer: F", num_options=4).status is ParseStatus.OUT_OF_RANGE


@pytest.mark.parametrize(
    "raw",
    [
        "I think the answer is probably A",
        "A because the shape rotates",
        "The answer is clearly B because of the rotation",
        "Option A is correct",
        "A is the answer",
        "The correct option is the first one",
        "Answer:",
        "The answer is",
        "Answer: AB",
        "Answer: rotation",
        "Answer: 2",
        "Answer: 1",
        "42",
        "1",
        "?",
        "None",
        "Answer: either A or B is fine, I cannot tell",
        "Options A, B, C and D are all plausible.",
        "A fine rotation of the shape results in D",
        "Answer choices:",
        "Ａ",
    ],
)
def test_malformed_output_never_selects_a_letter(raw):
    parsed = parse_answer(raw)
    assert parsed.status is ParseStatus.MALFORMED
    assert parsed.index is None
    assert not parsed.is_valid


def test_raw_output_is_preserved_exactly():
    raw = "  \nAnswer: b  \n"
    parsed = parse_answer(raw)
    assert parsed.raw is raw
    assert parse_answer(None).raw is None
    assert parse_answer("Answer: A or B").raw == "Answer: A or B"


def test_result_is_immutable():
    parsed = parse_answer("A")
    with pytest.raises(FrozenInstanceError):
        parsed.index = 3  # type: ignore[misc]


@pytest.mark.parametrize("raw", [b"A", 0, 1.5, ["A"]])
def test_non_text_output_is_a_programming_error(raw):
    with pytest.raises(TypeError):
        parse_answer(raw)  # type: ignore[arg-type]


@pytest.mark.parametrize("num_options", [0, 1, 27, -4])
def test_invalid_option_count_is_rejected(num_options):
    with pytest.raises(ValueError, match="num_options"):
        parse_answer("A", num_options=num_options)


@pytest.mark.parametrize("num_options", [True, 4.0, "4"])
def test_option_count_must_be_an_int(num_options):
    with pytest.raises(TypeError):
        parse_answer("A", num_options=num_options)  # type: ignore[arg-type]


def test_is_correct_requires_a_valid_matching_parse():
    assert is_correct(parse_answer("B"), 1)
    assert not is_correct(parse_answer("B"), 2)
    assert not is_correct(parse_answer("A or B"), 0)
    assert not is_correct(parse_answer(None), 0)
    assert not is_correct(ParsedAnswer("x", ParseStatus.MALFORMED, index=0), 0)


def test_scoring_keeps_invalid_outputs_in_the_denominator():
    outputs = ["A", "Answer: B", None, "A or B", "E", "I think C", "D"]
    answers = [0, 1, 2, 0, 0, 2, 0]
    report = score_outputs(outputs, answers)
    assert report.total == 7
    assert report.correct == 2
    assert report.invalid == 4  # unparseable outputs, the wrong valid answer is separate
    assert report.total - report.correct == 5  # every non-correct item is a miss
    assert report.accuracy == pytest.approx(2 / 7)
    assert report.status_counts == {
        "valid": 3,
        "missing": 1,
        "ambiguous": 1,
        "out_of_range": 1,
        "malformed": 1,
    }
    assert sum(report.status_counts.values()) == report.total


def test_all_invalid_outputs_score_zero_not_undefined():
    report = score_outputs([None, "", "maybe"], [0, 1, 2])
    assert report.total == 3
    assert report.correct == 0
    assert report.accuracy == 0.0
    assert report.invalid == 3


def test_scoring_validates_its_inputs():
    with pytest.raises(ValueError, match="2 outputs for 3 answers"):
        score_outputs(["A", "B"], [0, 1, 2])
    with pytest.raises(ValueError, match="empty"):
        score_outputs([], [])
    with pytest.raises(ValueError, match="answer index"):
        score_outputs(["A"], [4])
    with pytest.raises(ValueError, match="answer index"):
        score_outputs(["A"], [-1])
    with pytest.raises(ValueError, match="answer index"):
        score_outputs(["A"], [True])
    with pytest.raises(ValueError, match="answer index"):
        score_outputs(["A"], [1.0])  # type: ignore[list-item]


def test_scoring_honours_the_option_count():
    report = score_outputs(["E", "Answer: F"], [4, 5], num_options=6)
    assert report.correct == 2
    assert score_outputs(["E"], [3]).status_counts["out_of_range"] == 1


def test_parser_labels_match_the_dataset_labels():
    assert answers_module._LABELS.startswith(OPTION_LABELS)


@pytest.mark.parametrize("name", ["tetris2d", "tetris3d"])
def test_dataset_answer_labels_round_trip(name):
    cfg = DatasetConfig.from_json(ROOT / "configs" / "dataset" / f"{name}.json")
    split = cfg.splits[0]
    seen = set()
    for index in range(12):
        sample = generate_sample(cfg, split, index)
        seen.add(sample["answer"])
        for raw in (sample["answer_label"], f"Answer: {sample['answer_label'].lower()}"):
            assert parse_answer(raw, cfg.num_options).index == sample["answer"]
        wrong = OPTION_LABELS[(sample["answer"] + 1) % cfg.num_options]
        assert not is_correct(parse_answer(wrong, cfg.num_options), sample["answer"])
    assert len(seen) > 1


def test_fuzzed_text_never_raises_and_valid_results_are_grounded():
    rng = DetRng("answer-parser-fuzz")
    alphabet = list("ABCDEabcde  \n\t.,:;()[]*_`'\"/&|-0123") + [
        "Answer",
        "answer:",
        "is",
        "the",
        "or",
        "and",
    ]
    for _ in range(4000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        parsed = parse_answer(raw)
        assert parsed.raw == raw
        if parsed.is_valid:
            assert parsed.index in range(4)
            assert re.search(parsed.label, raw, flags=re.IGNORECASE)
        else:
            assert parsed.index is None
            assert parsed.label is None


def test_prose_without_a_declaration_is_never_valid():
    rng = DetRng("answer-parser-prose")
    words = ["the", "shape", "rotates", "so", "option", "A", "b", "C", "d", "is", "a", "plausible", "result"]
    for _ in range(2000):
        sentence = " ".join(rng.choice(words) for _ in range(rng.randint(3, 12))) + "."
        parsed = parse_answer(sentence)
        # Only text made entirely of labels and filler words can be a declaration.
        only_labels = all(w.lower() in {"option", "a", "b", "c", "d"} for w in sentence.rstrip(".").split())
        if not only_labels:
            assert not parsed.is_valid, sentence


def test_eval_modules_do_not_import_model_or_image_libraries():
    source = (ROOT / "src" / "wm_vlm" / "eval" / "answers.py").read_text()
    imported = set(re.findall(r"^(?:from|import)\s+([A-Za-z_][\w]*)", source, flags=re.MULTILINE))
    assert imported <= {"__future__", "re", "collections", "dataclasses", "enum"}


def test_documented_example_runs_and_its_forms_hold():
    doc = (ROOT / "docs" / "answer_parsing.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", doc, flags=re.DOTALL)
    assert len(blocks) == 1
    exec(compile(blocks[0], "docs/answer_parsing.md", "exec"), {})
    # Every accepted form in the documented table really parses to B.
    for raw in ["Answer: B", "answer b", "Final answer: (B)", "The answer is B.", "Correct answer: option b"]:
        assert parse_answer(raw).index == 1, raw
    for raw in ["B", "b", "(B)", "[B]", "B.", "B)", "Option B"]:
        assert parse_answer(raw).index == 1, raw
