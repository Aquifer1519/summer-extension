"""
Prototype: AI-generated text detection using a pretrained classifier
fine-tuned on the HC3 dataset (Human ChatGPT Comparison Corpus).

IMPORTANT CONTEXT: HC3-based detectors are trained on long-form Q&A
responses (formal, paragraph-length). Your target text -- short political
X/Twitter posts -- is a different domain (short, informal, hashtag-heavy,
opinionated rather than explanatory). This script is deliberately built to
surface that mismatch, not assume it away. Don't trust the accuracy numbers
here until you've run this against real examples from your actual domain.
"""

from transformers import pipeline

detector_pipe = pipeline(
    "text-classification",
    model="Hello-SimpleAI/chatgpt-detector-roberta",
    top_k=None,
)

# TODO: confirm these against the raw output of detector_pipe() on a test
# string -- placeholder guesses based on the model's typical convention.
HUMAN_LABEL = "Human"
AI_LABEL = "ChatGPT"


def detect(text: str):
    raw = detector_pipe(text)[0]
    by_label = {r["label"]: r["score"] for r in raw}

    return {
        "text": text,
        "ai_likelihood": round(by_label.get(AI_LABEL, float("nan")), 4),
        "raw_scores": {k: round(v, 4) for k, v in by_label.items()},
    }


if __name__ == "__main__":
    # A mix of known-human and synthetic known-AI examples, all in a
    # political-post style, to stress-test the domain mismatch.
    # NOTE: the "AI-generated" examples below are text I (Claude) wrote
    # to resemble typical LLM output style -- not pulled from any real
    # AI-generated dataset. Treat this as a rough smoke test, not a
    # rigorous eval.

    human_examples = [
        # Pulled-style human tweets (short, informal, hashtag-heavy,
        # opinionated) -- similar to what you saw in the election dataset.
        "Trump will stay! #Trump2020 #ElectionNight #Election2020",
        "Not according to the REPUTABLE polling sources. Nobody wants Trump re-elected. #VotingForBiden",
        "Did your vote help people or hurt people? We are not perfect people, but if we continue to hate each other, this country won't be a better place.",
    ]

    ai_examples = [
        # Deliberately written in a more formal, hedged, structurally
        # balanced style typical of LLM output -- useful for testing
        # whether the detector picks up on register at all, even though
        # real AI-generated political posts would likely be prompted to
        # sound more like the human examples above.
        "It is important to consider multiple perspectives on this election. Both candidates have proposed policies that could significantly impact the economy and healthcare system.",
        "While opinions on this topic vary widely, it's worth noting that voter turnout has historically played a crucial role in determining election outcomes.",
    ]

    print("=== First, confirm label names ===")
    print(detector_pipe(human_examples[0])[0])
    print()

    print("=== Human-written examples ===")
    for ex in human_examples:
        result = detect(ex)
        print(f"\nTEXT: {result['text']}")
        print(f"  AI likelihood: {result['ai_likelihood']}")
        print(f"  Raw: {result['raw_scores']}")

    print("\n=== AI-style examples (formal, hedged register) ===")
    for ex in ai_examples:
        result = detect(ex)
        print(f"\nTEXT: {result['text']}")
        print(f"  AI likelihood: {result['ai_likelihood']}")
        print(f"  Raw: {result['raw_scores']}")
