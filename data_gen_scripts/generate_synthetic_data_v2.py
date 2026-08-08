"""
Generate diverse synthetic AI-written political-tweet-style text.

Output remains compatible with the existing training pipeline:
synthetic_tweets.json -> [{"text": "...", "label": "ai"}, ...]

V2 changes:
- Uses concrete situations + motivations instead of over-specifying six style axes.
- Explicitly varies tweet length.
- Generates reactions, questions, jokes, personal observations, arguments,
  information sharing, and low-information posts.
- Makes emojis/mentions/hashtags optional features rather than required formats.
- Adds a lightweight local quality filter.
- Retries malformed batches instead of immediately discarding them.
- Keeps generation intentionally diverse so the detector does not learn one
  narrow prompt/style signature.
"""

import json
import random
import re
import time
from collections import Counter

from openai import OpenAI

client = OpenAI()

MODEL = "gpt-4o-mini"
OUTPUT_FILE = "synthetic_tweets.json"
NUM_TWEETS_TO_GENERATE = 300
TWEETS_PER_API_CALL = 10

# Broad topics. These are deliberately less prescriptive than the original
# generator; the situation/context determines how the topic appears in a post.
TOPICS = [
    "the economy and inflation",
    "gas prices",
    "housing affordability",
    "student loans",
    "minimum wage policy",
    "tax policy",
    "unemployment and the job market",
    "government spending and the national debt",
    "healthcare",
    "prescription drug prices",
    "the Affordable Care Act",
    "abortion and reproductive rights",
    "mental healthcare access",
    "immigration",
    "border security",
    "asylum and refugees",
    "citizenship policy",
    "climate change",
    "energy policy",
    "electric vehicles",
    "natural disasters and government response",
    "American democracy",
    "voting and election administration",
    "gerrymandering",
    "campaign spending",
    "voter ID",
    "mail-in and early voting",
    "the Supreme Court",
    "judicial appointments",
    "criminal justice",
    "policing",
    "foreign policy",
    "US-China relations",
    "an overseas conflict",
    "NATO",
    "trade and tariffs",
    "gun policy",
    "education",
    "school curriculum and book bans",
    "LGBTQ-related policy debates",
    "social media regulation",
    "free speech online",
    "big tech and antitrust",
    "labor unions",
    "homelessness",
    "a political debate",
    "political media coverage",
    "political polarization",
    "a politician's gaffe or scandal",
    "a political advertisement",
    "third-party candidates",
    "local or state politics",
    "a congressional hearing",
    "a presidential press conference",
]

STANCES = [
    "supportive of the Democratic position",
    "supportive of the Republican position",
    "critical of both major parties",
    "supportive of an independent or third-party position",
    "skeptical and not firmly committed to either side",
    "mostly observational rather than partisan",
]

MOTIVATIONS = [
    "reacting emotionally to something they just saw",
    "complaining about something",
    "asking a genuine question",
    "trying to persuade another user",
    "disagreeing with another user",
    "agreeing with another user",
    "making a joke or sarcastic observation",
    "sharing a piece of information",
    "correcting or challenging a claim",
    "describing a personal reaction",
    "venting without trying to make a polished argument",
    "posting a quick observation without much explanation",
]

CONTEXTS = [
    "a person just saw a political headline",
    "a person is watching a televised political debate",
    "a person is scrolling through political posts",
    "a person heard a politician make a controversial statement",
    "a person saw a clip from a news conference",
    "a person is reacting to a recent vote or court decision",
    "a person is talking about something affecting their household",
    "a person is discussing politics with another user",
    "a person saw a political advertisement",
    "a person is reacting to a local political issue",
    "a person is sharing something they just learned",
    "a person is frustrated by an argument they keep seeing online",
    "a person has only read a headline and is asking what it means",
    "a person is reacting to an election result",
    "a person is commenting on a politician's public statement",
]

LENGTHS = [
    "very short, roughly 5-30 characters",
    "short, roughly 30-70 characters",
    "short-to-medium, roughly 70-120 characters",
    "medium, roughly 120-180 characters",
    "longer, roughly 180-260 characters",
]

# Optional surface features. They are probabilities, not mutually exclusive
# "formats", so the dataset does not mechanically associate one feature with
# one type of tweet.
OPTIONAL_FEATURES = [
    "no special formatting",
    "possibly include one emoji if it naturally fits",
    "possibly include one hashtag if it naturally fits",
    "possibly mention another user if it naturally fits",
    "casual lowercase typing is acceptable",
    "a small typo or missing punctuation is acceptable",
    "a sentence fragment is acceptable",
    "normal polished punctuation is acceptable",
]

# These are not forbidden words. They are used only as weak signals in the
# local quality filter because repeated boilerplate can make the AI class
# artificially easy to detect.
BOILERPLATE_PATTERNS = [
    r"\bit'?s worth noting\b",
    r"\bconstructive (?:discussion|discussions)\b",
    r"\bmoving forward\b",
    r"\breal solutions\b",
    r"\bwhat matters\b",
    r"\burgent issues?\b",
    r"\bnational security concerns\b",
    r"\bboth sides need to\b",
    r"\bthis is not about .* it'?s about\b",
    r"\bcan we get some real leadership\b",
]

def random_spec():
    return {
        "topic": random.choice(TOPICS),
        "stance": random.choice(STANCES),
        "motivation": random.choice(MOTIVATIONS),
        "context": random.choice(CONTEXTS),
        "length": random.choice(LENGTHS),
        "feature": random.choice(OPTIONAL_FEATURES),
    }


def build_prompt():
    specs = [random_spec() for _ in range(TWEETS_PER_API_CALL)]

    spec_lines = "\n".join(
        f"{i + 1}. Topic: {s['topic']} | "
        f"Context: {s['context']} | "
        f"Motivation: {s['motivation']} | "
        f"Stance: {s['stance']} | "
        f"Length: {s['length']} | "
        f"Surface detail: {s['feature']}"
        for i, s in enumerate(specs)
    )

    return f"""
Write {TWEETS_PER_API_CALL} plausible social-media posts about US politics.

Each numbered post has its OWN situation. Follow its situation, but do not
force every detail into the post. A real person would naturally decide what
to mention and what to leave out.

{spec_lines}

Important:
- Each post must be under 280 characters.
- Do not make every post an argument or mini-essay.
- Some posts should be questions, reactions, jokes, observations, complaints,
  replies, information sharing, or incomplete/low-information thoughts.
- Do not make every post polished. Some may be casual, fragmentary, lowercase,
  lightly typoed, or loosely punctuated, but most should still be plausible.
- Do not force emojis, hashtags, mentions, rhetorical questions, or slang.
  Use them only when they naturally fit the situation.
- Avoid generic essay, policy-memo, press-release, or news-report language.
- Avoid stock AI-style conclusions and repetitive openings.
- Do not use the same sentence structure across multiple posts.
- Do not explain the writing process or mention these instructions.
- Do not invent a specific current event unless the situation provides one.
- Political viewpoints should vary naturally. Do not make every post strongly
  partisan.
- Do not deliberately imitate a named real person or public figure.
- The goal is ordinary-looking variety, not exaggerated "human-like" quirks.

Return ONLY a JSON array of exactly {TWEETS_PER_API_CALL} strings.
No markdown fences. No commentary.
"""


def parse_batch(raw_text):
    raw_text = (raw_text or "").strip()

    # First try the response exactly as returned.
    try:
        tweets = json.loads(raw_text)
        if isinstance(tweets, list):
            return [str(t) for t in tweets]
    except (json.JSONDecodeError, ValueError):
        pass

    # Handle accidental markdown fences or surrounding commentary.
    cleaned = raw_text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        tweets = json.loads(cleaned.strip())
        if isinstance(tweets, list):
            return [str(t) for t in tweets]
    except (json.JSONDecodeError, ValueError):
        pass

    # Last-resort extraction of the JSON array.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            tweets = json.loads(cleaned[start:end + 1])
            if isinstance(tweets, list):
                return [str(t) for t in tweets]
        except json.JSONDecodeError:
            pass

    return None


def generate_batch(max_retries=3):
    prompt = build_prompt()

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=3000,
                temperature=1.05,
                frequency_penalty=0.5,
                presence_penalty=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.choices[0].message.content or ""
            tweets = parse_batch(raw_text)

            if tweets is not None and tweets:
                return tweets

            # Preserve the raw response for debugging.
            with open("failed_batches.log", "a", encoding="utf-8") as f:
                f.write("=" * 40 + "\n")
                f.write(f"Attempt {attempt}/{max_retries}\n")
                f.write(raw_text + "\n")

            if attempt < max_retries:
                time.sleep(1.0 * attempt)

        except Exception as exc:
            with open("failed_batches.log", "a", encoding="utf-8") as f:
                f.write("=" * 40 + "\n")
                f.write(f"API error on attempt {attempt}/{max_retries}: {exc}\n")

            if attempt < max_retries:
                time.sleep(1.0 * attempt)

    return None


def normalize_tweet(text):
    return re.sub(r"\s+", " ", text.strip())


def quality_flags(text):
    """
    Lightweight deterministic checks.

    These are intentionally conservative. We want to remove obvious bad
    generations without making the AI class unnaturally "perfect".
    """
    text = normalize_tweet(text)

    if not text:
        return ["empty"]

    if len(text) > 280:
        return ["over_280"]

    if "@someuser" in text.lower():
        return ["placeholder_handle"]

    flags = []

    if len(text) < 4:
        flags.append("too_short")

    # Detect obvious instruction leakage.
    leakage = [
        "here is the tweet",
        "here's the tweet",
        "as an ai",
        "according to the prompt",
        "topic:",
        "stance:",
        "persona:",
        "writing approach:",
    ]
    lower = text.lower()
    if any(x in lower for x in leakage):
        flags.append("instruction_leak")

    # Excessive formatting often indicates the model ignored the tweet format.
    if text.count("#") > 4:
        flags.append("too_many_hashtags")

    if text.count("@") > 3:
        flags.append("too_many_mentions")

    # Very long sequences of repeated punctuation are usually malformed output.
    if re.search(r"[!?]{5,}", text):
        flags.append("punctuation_spam")

    # Weak signal for boilerplate. We don't reject one occurrence.
    boilerplate_hits = sum(
        bool(re.search(pattern, lower)) for pattern in BOILERPLATE_PATTERNS
    )
    if boilerplate_hits >= 2:
        flags.append("boilerplate")

    return flags


def accept_tweet(text, seen):
    text = normalize_tweet(text)

    if text in seen:
        return False

    flags = quality_flags(text)
    if flags:
        return False

    return True


def main():
    all_tweets = []
    seen = set()

    api_failures = 0
    rejected_outputs = 0
    batch_count = 0

    while len(all_tweets) < NUM_TWEETS_TO_GENERATE:
        batch = generate_batch()
        batch_count += 1

        if batch is None:
            api_failures += 1
            print(
                f"  Warning: batch failed after retries "
                f"({len(all_tweets)}/{NUM_TWEETS_TO_GENERATE})"
            )
            continue

        new_count = 0

        for raw_tweet in batch:
            if not isinstance(raw_tweet, str):
                rejected_outputs += 1
                continue

            tweet = normalize_tweet(raw_tweet)

            if not accept_tweet(tweet, seen):
                rejected_outputs += 1
                continue

            seen.add(tweet)
            all_tweets.append({
                "text": tweet,
                "label": "ai",
            })
            new_count += 1

            if len(all_tweets) >= NUM_TWEETS_TO_GENERATE:
                break

        print(
            f"  Batch added {new_count} new tweets "
            f"(total: {len(all_tweets)}/{NUM_TWEETS_TO_GENERATE})"
        )

        time.sleep(0.5)

    all_tweets = all_tweets[:NUM_TWEETS_TO_GENERATE]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_tweets, f, indent=2, ensure_ascii=False)

    lengths = [len(item["text"]) for item in all_tweets]
    hashtags = sum("#" in item["text"] for item in all_tweets)
    mentions = sum("@" in item["text"] for item in all_tweets)
    emojis = sum(
        any(ord(ch) > 0x1F000 for ch in item["text"])
        for item in all_tweets
    )

    print(f"\nWrote {len(all_tweets)} synthetic AI tweets to {OUTPUT_FILE}")
    print(f"API/batch failures after retries: {api_failures}")
    print(f"Rejected individual outputs: {rejected_outputs}")
    print(
        f"Length: min={min(lengths)}, "
        f"median={sorted(lengths)[len(lengths)//2]}, "
        f"max={max(lengths)}"
    )
    print(f"Tweets containing hashtags: {hashtags}/{len(all_tweets)}")
    print(f"Tweets containing mentions: {mentions}/{len(all_tweets)}")
    print(f"Tweets containing likely emoji: {emojis}/{len(all_tweets)}")


if __name__ == "__main__":
    main()
