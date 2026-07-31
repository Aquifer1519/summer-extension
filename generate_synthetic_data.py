"""
Generate a diverse batch of synthetic AI-written political-tweet-style
text, to use as the "AI" class when fine-tuning the detector -- paired
against real human tweets from the election dataset you already have.

Requires an OpenAI API key:
    pip install openai
    export OPENAI_API_KEY=your_key_here   # (Windows: set OPENAI_API_KEY=...)

Usage:
    python generate_synthetic_data.py

Output: synthetic_tweets.json -- a list of {"text": ..., "label": "ai"}
objects, ready to be combined with a sample of real human tweets into a
single training file for fine_tune_detector.py.

DESIGN NOTE ON DIVERSITY: generating N completions from one fixed prompt
tends to produce near-duplicate style/structure -- bad for training a
classifier, since it'd learn to recognize "one AI's one style" rather than
AI-generated text in general. This script varies topic, stance, tone, and
format instructions across calls to reduce that.
"""

import json
import random
import time

from openai import OpenAI

client = OpenAI()  # picks up OPENAI_API_KEY from environment

MODEL = "gpt-4o-mini"  # confirm this matches a model your account has access to
OUTPUT_FILE = "synthetic_tweets.json"
NUM_TWEETS_TO_GENERATE = 500
TWEETS_PER_API_CALL = 10  # ask for a batch per call, cheaper than one-at-a-time

TOPICS = [
    "the economy and inflation",
    "healthcare policy",
    "immigration policy",
    "climate change and energy policy",
    "the state of American democracy",
    "voting and election integrity",
    "the Supreme Court",
    "foreign policy and international relations",
    "gun policy",
    "education policy",
    "a recent political debate",
    "campaign fundraising and political spending",
    "media coverage of politics",
    "a swing state election result",
    "political polarization in the US",
]

STANCES = [
    "supportive of the Democratic position",
    "supportive of the Republican position",
    "critical of both major parties",
    "focused on a third-party or independent perspective",
    "skeptical and questioning, not taking a firm side",
]

TONES = [
    "angry and combative",
    "sarcastic",
    "earnest and concerned",
    "celebratory",
    "matter-of-fact, reporting information",
    "mocking or dismissive of the opposing view",
]

FORMATS = [
    "using several hashtags",
    "with no hashtags at all, just plain text",
    "written as a reply to another user (include an @mention placeholder like @someuser)",
    "including an emoji or two",
    "as a short, punchy one-liner",
    "as a slightly longer, multi-sentence post",
]


def build_prompt() -> str:
    topic = random.choice(TOPICS)
    stance = random.choice(STANCES)
    tone = random.choice(TONES)
    fmt = random.choice(FORMATS)

    return f"""Write {TWEETS_PER_API_CALL} short social media posts (tweet-length, under 280 characters each) about {topic}.
Each post should be {stance}, with a {tone} tone, {fmt}.
Vary the wording, structure, and specific claims across all {TWEETS_PER_API_CALL} posts -- they should not read like variations of the same sentence.
Write them the way real, opinionated people post about politics online -- direct, informal, not neutral-sounding.

Return ONLY a JSON array of {TWEETS_PER_API_CALL} strings, nothing else -- no preamble, no markdown code fences.
Example format: ["post one text here", "post two text here", ...]"""


def generate_batch() -> list[str]:
    prompt = build_prompt()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = response.choices[0].message.content.strip()

    # Defensive parsing -- strip accidental code fences if the model adds them
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        tweets = json.loads(raw_text)
        if not isinstance(tweets, list):
            raise ValueError("Expected a JSON list")
        return [str(t) for t in tweets]
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  Warning: failed to parse batch response, skipping. Error: {e}")
        print(f"  Raw response was: {raw_text[:200]}...")
        return []


def main():
    all_tweets = []
    seen = set()  # dedupe exact repeats across batches

    while len(all_tweets) < NUM_TWEETS_TO_GENERATE:
        batch = generate_batch()
        new_count = 0
        for t in batch:
            if t not in seen:
                seen.add(t)
                all_tweets.append({"text": t, "label": "ai"})
                new_count += 1

        print(
            f"  Batch added {new_count} new tweets (total: {len(all_tweets)}/{NUM_TWEETS_TO_GENERATE})"
        )
        time.sleep(0.5)  # light rate-limit courtesy pause

    all_tweets = all_tweets[:NUM_TWEETS_TO_GENERATE]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_tweets, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(all_tweets)} synthetic AI tweets to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
