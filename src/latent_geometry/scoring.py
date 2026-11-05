"""SPBench-SI direct-answer scoring, matching SpatialLadder's VSI utilities.

Reference: ZJU-REAL/SpatialLadder/eval_spld/data_utils/vsi_utils.py.
Keep the upstream numerical parsing
and threshold construction for comparable scores, including their quirks.
"""
import math
import re

import numpy as np


MC_TYPES = {"object_rel_direction", "object_rel_distance"}
NUMERIC_TYPES = {"object_abs_distance", "object_size_estimation"}


def parse_number(text):
    text = text.strip().lower()
    words = ("one two three four five six seven eight nine ten eleven twelve thirteen "
             "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()
    numbers = {word: index for index, word in enumerate(words, 1)}
    numbers.update(dict(zip("thirty forty fifty sixty seventy eighty ninety".split(), range(30, 100, 10))))
    numbers.update({"zero": 0, "a": 1, "an": 1})
    for word, value in numbers.items():
        if re.search(r"\b" + word + r"\b", text):
            return float(value)
    match = re.search(r"(\d+(\.\d+)?)", text)
    return float(match[0]) if match else None


def score_spbench(prediction, record):
    kind, reference = record["question_type"], record["answer"]
    if kind in MC_TYPES:
        prediction = prediction.replace("Answer:", "")
        first = prediction.split(" ")[0].strip()
        parsed = first.rstrip(".") if re.fullmatch(r"[A-D]\.?", first) else prediction.strip()
        return float(parsed.lower() == reference.lower())
    if kind not in NUMERIC_TYPES:
        raise ValueError(f"Unsupported SPBench-SI question type: {kind}")
    target = float(reference)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("SPBench numerical references must be finite and positive")
    predicted = parse_number(prediction)
    if predicted is None or not math.isfinite(predicted):
        return 0.0
    # Match upstream linspace construction exactly (rather than arange rounding).
    thresholds = np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05 + 2))
    return float((abs(predicted - target) / target <= 1 - thresholds).mean())


def summarize_spbench(rows):
    by_type = {}
    for row in rows:
        group = by_type.setdefault(row["question_type"], {"count": 0, "score_sum": 0.0})
        group["count"] += 1
        group["score_sum"] += row["score"]
    for kind, group in by_type.items():
        group["score"] = group.pop("score_sum") / group["count"]
        group["metric"] = "accuracy" if kind in MC_TYPES else "MRA"
    def average(kinds):
        scores = [group["score"] for kind, group in by_type.items() if kind in kinds]
        return sum(scores) / len(scores) if scores else None
    return {"by_question_type": by_type, "overall_score": average(MC_TYPES | NUMERIC_TYPES),
            "mcq_accuracy": average(MC_TYPES), "numerical_mra": average(NUMERIC_TYPES),
            "overall_aggregation": "Unweighted mean of task scores; all scores in [0, 1]"}
