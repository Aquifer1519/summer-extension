"""
Prototype: continuous emotional polarity scale (strongly negative -> strongly
positive), using a single pretrained 3-class sentiment model.

Rather than a discrete label (positive/negative/neutral) or a separate
7-emotion model, we derive a continuous score directly from the model's
class probabilities:

    score = P(positive) - P(negative)

Range: -1.0 (strongly negative) to +1.0 (strongly positive), with values
near 0 meaning neutral or mixed. This also avoids the earlier problem of
two independently-trained models (polarity + emotion) contradicting each
other on the same text.
"""

from transformers import pipeline

# top_k=None returns scores for ALL classes (negative/neutral/positive),
# not just the winning label -- we need all three to compute the scale.
polarity_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None,
)


# Human-readable buckets for the continuous score. Tune these thresholds
# once you've eyeballed enough real examples -- these are a reasonable
# starting guess, not derived from data.
def label_for_score(score: float) -> str:
    if score <= -0.6:
        return "strongly negative"
    elif score <= -0.2:
        return "negative"
    elif score < 0.2:
        return "neutral"
    elif score < 0.6:
        return "positive"
    else:
        return "strongly positive"


def analyze(text: str):
    scores = polarity_pipe(text)[
        0
    ]  # list of {"label": ..., "score": ...} for all 3 classes
    by_label = {s["label"]: s["score"] for s in scores}

    p_pos = by_label.get("positive", 0.0)
    p_neg = by_label.get("negative", 0.0)
    scale_score = p_pos - p_neg

    return {
        "text": text,
        "scale_score": round(scale_score, 4),
        "scale_label": label_for_score(scale_score),
        "raw_scores": {k: round(v, 4) for k, v in by_label.items()},
    }


if __name__ == "__main__":
    samples = [
        "I absolutely love this, best day of my life!",
        "This is the worst customer service experience I've ever had.",
        "The meeting is scheduled for 3pm on Tuesday.",
        "I'm so nervous about the interview tomorrow, I can barely sleep.",
    ]

    for s in samples:
        result = analyze(s)
        print(f"\nTEXT: {result['text']}")
        print(f"  Scale: {result['scale_score']:+.4f}  ({result['scale_label']})")
        print(f"  Raw scores: {result['raw_scores']}")
