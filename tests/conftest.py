import math

import pytest
from datasets import Dataset


class WhitespaceTokenizer:
    """Offline stand-in for the Qwen tokenizer: one token per whitespace-separated word."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "<sys> default system prompt </sys> " + " ".join(f"<{m['role']}> {m['content']} </end>" for m in messages)
        return text + (" <assistant>" if add_generation_prompt else "")

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}


def make_row(prompt, chosen, rejected, cr, rr, source="src", cm="model-a", rm="model-b", user_turn=None):
    user = prompt if user_turn is None else user_turn
    return {
        "source": source, "prompt": prompt,
        "chosen": [{"role": "user", "content": user}, {"role": "assistant", "content": chosen}],
        "chosen-rating": cr, "chosen-model": cm,
        "rejected": [{"role": "user", "content": user}, {"role": "assistant", "content": rejected}],
        "rejected-rating": rr, "rejected-model": rm,
    }


# (row, expected drop reason or None). Indices are stable: tests refer to them by position.
SPECIAL_ROWS = [
    (make_row("nan rating", "a b", "c", math.nan, 2.0), "malformed_rating"),
    (make_row("none rating", "a b", "c", None, 2.0), "malformed_rating"),
    (make_row("out of range", "a b", "c", 6.0, 2.0), "malformed_rating"),
    (make_row("mismatched user turn", "a b", "c", 5.0, 2.0, user_turn="other"), "malformed_structure"),
    (make_row("   ", "a b", "c", 5.0, 2.0), "empty_prompt"),
    (make_row("empty chosen", "", "c", 5.0, 2.0), "empty_chosen"),
    (make_row("blank rejected", "a b", " \n\t ", 5.0, 2.0), "empty_rejected"),
    (make_row("identical", "same answer", " same answer ", 5.0, 2.0), "identical_responses"),
    (make_row("tie", "a b", "c", 3.0, 3.0), "non_positive_margin"),
    (make_row("inverted", "a b", "c", 2.0, 4.0), "non_positive_margin"),
    (make_row("overlong", " ".join(["w"] * 200), "c", 5.0, 2.0), "overlong"),
    (make_row("Duplicate   Prompt", "first kept", "x", 5.0, 1.0), None),
    (make_row("duplicate prompt", "second dropped", "y", 4.0, 1.0), "duplicate_normalized_prompt"),
    (make_row("DUPLICATE PROMPT ", "third dropped", "z", 4.5, 1.0), "duplicate_normalized_prompt"),
]


def build_rows(n_valid: int = 400):
    rows = [r for r, _ in SPECIAL_ROWS]
    sources = ["alpha", "beta", "gamma"]
    for i in range(n_valid):
        cr = 1.25 + 0.25 * ((i * 7) % 16)
        rr = 1.0 + 0.25 * ((i * 3) % ((cr - 1.0) / 0.25)) if cr > 1.0 else 1.0
        rows.append(make_row(f"valid prompt number {i}", "good " * (1 + i % 9), "bad " * (1 + i % 5), cr, rr,
                             source=sources[i % 3], cm=f"m{i % 4}", rm=f"m{(i + 1) % 4}"))
    return rows


@pytest.fixture
def raw_dataset():
    return Dataset.from_list(build_rows())


@pytest.fixture
def tokenizer():
    return WhitespaceTokenizer()


@pytest.fixture
def prep_cfg():
    return {
        "max_total_tokens": 100,
        "split": {"seed": 0, "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1}},
        "buckets": {"quantiles": [1 / 3, 2 / 3]},
    }
