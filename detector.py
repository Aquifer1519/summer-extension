"""
AI-generated text detection -- now using the custom fine-tuned model
(trained via custom_detector.py) instead of the pretrained HC3 classifier.

detect(text) keeps the same return shape as before ({"text", "ai_likelihood",
"raw_scores"}), so app.py and anything else importing from here needs no
changes -- this file was always the intended swap point.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./tweepfake-detector"  # matches OUTPUT_DIR in custom_detector.py

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()  # inference mode -- disables dropout etc.

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Matches the label mapping used during training in custom_detector.py:
# HUMAN_VALUE -> 0, AI_VALUE -> 1. The saved model's config doesn't know
# these human-readable names on its own (Trainer/RobertaForSequenceClassification
# defaults to generic "LABEL_0"/"LABEL_1"), so we map manually here.
ID_TO_LABEL = {0: "human", 1: "ai"}


def detect(text: str):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,  # matches MAX_LENGTH used during training
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    by_label = {ID_TO_LABEL[i]: probs[i].item() for i in range(len(probs))}

    return {
        "text": text,
        "ai_likelihood": round(by_label.get("ai", float("nan")), 4),
        "raw_scores": {k: round(v, 4) for k, v in by_label.items()},
    }


if __name__ == "__main__":
    # Same test set as before, plus the real human tweet that the OLD
    # pretrained detector confidently (and wrongly) called 99.9% AI --
    # worth re-checking that one specifically against the new model.
    test_examples = [
        "Trump will stay! #Trump2020 #ElectionNight #Election2020",
        "It is important to consider multiple perspectives on this election. Both candidates have proposed policies that could significantly impact the economy and healthcare system.",
        "Did your vote help people or hurt people? We are not perfect people, but if we continue to hate each other, this country won't be a better place.",
    ]

    for ex in test_examples:
        result = detect(ex)
        print(f"\nTEXT: {result['text']}")
        print(f"  AI likelihood: {result['ai_likelihood']}")
        print(f"  Raw: {result['raw_scores']}")
