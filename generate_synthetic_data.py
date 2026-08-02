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
NUM_TWEETS_TO_GENERATE = 10000
TWEETS_PER_API_CALL = 10  # ask for a batch per call, cheaper than one-at-a-time

TOPICS = [
    # Economy
    "the economy and inflation",
    "gas prices",
    "the housing market and affordability",
    "student loan debt and forgiveness",
    "minimum wage policy",
    "tax policy for the wealthy vs middle class",
    "unemployment and job market conditions",
    "the national debt and government spending",
    # Healthcare
    "healthcare policy",
    "prescription drug prices",
    "the Affordable Care Act / ACA",
    "abortion policy and reproductive rights",
    "mental health care access",
    # Immigration
    "immigration policy",
    "border security",
    "asylum seekers and refugees",
    "pathway to citizenship debates",
    # Environment / energy
    "climate change and energy policy",
    "renewable energy vs fossil fuels",
    "electric vehicle policy and incentives",
    "natural disasters and government response",
    # Democracy / elections
    "the state of American democracy",
    "voting and election integrity",
    "gerrymandering and redistricting",
    "campaign fundraising and political spending",
    "a swing state election result",
    "voter ID laws",
    "mail-in and early voting",
    # Courts / law
    "the Supreme Court",
    "a recent Supreme Court ruling",
    "judicial appointments and confirmations",
    "criminal justice reform",
    "policing and law enforcement policy",
    # Foreign policy
    "foreign policy and international relations",
    "US relations with China",
    "US involvement in an overseas conflict",
    "NATO and international alliances",
    "trade policy and tariffs",
    # Domestic social issues
    "gun policy",
    "education policy",
    "school curriculum and book bans",
    "LGBTQ rights and policy debates",
    "social media regulation and free speech online",
    "big tech and antitrust policy",
    "labor unions and workers' rights",
    "housing and homelessness policy",
    # Political process / media
    "a recent political debate",
    "media coverage of politics",
    "political polarization in the US",
    "a politician's public gaffe or scandal",
    "a political ad or campaign message",
    "third-party and independent candidates",
    "local or state-level politics",
    "a congressional hearing",
    "a presidential press conference",
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
    "neutral and hedged, presenting both sides journalistically (e.g. 'while supporters argue X, critics say Y')",
    "resigned or exhausted-sounding, like someone tired of the news cycle",
    "condescending, explaining something as if the reader doesn't already know it",
]

FORMATS = [
    "using several hashtags",
    "with no hashtags at all, just plain text",
    "written as a reply to another user (include an @mention placeholder like @someuser)",
    "including an emoji or two",
    "as a short, punchy one-liner",
    "as a slightly longer, multi-sentence post",
    "using modern internet meme phrasing, e.g. starting with 'not me...' or 'the way...' or 'bro...'",
    "all lowercase, no punctuation, casual texting style",
    "as a quote-tweet style reaction to something someone else supposedly said",
    "as the start of a longer thread (e.g. '1/' or 'thread:' at the start)",
]

PERSONAS = [
    "a partisan pundit account with a large following",
    "an ordinary person just venting after seeing the news",
    "a self-styled independent/centrist who criticizes both sides",
    "a young, terminally-online commenter",
    "an older, more formal social media user who isn't very tech-savvy",
    "a local news aggregator account sharing a headline with brief commentary",
    "someone quote-replying to argue with a stranger",
]


def build_prompt() -> str:
    # Assign a DISTINCT topic/stance/tone/format/persona combo to each
    # individual tweet in the batch -- assigning one combo per whole batch
    # (the old approach) causes all N tweets in a call to read like
    # paraphrases of each other, since they share the same instructions.
    specs = []
    for i in range(TWEETS_PER_API_CALL):
        specs.append(
            {
                "topic": random.choice(TOPICS),
                "stance": random.choice(STANCES),
                "tone": random.choice(TONES),
                "format": random.choice(FORMATS),
                "persona": random.choice(PERSONAS),
            }
        )

    spec_lines = "\n".join(
        f"{i+1}. Topic: {s['topic']} | Stance: {s['stance']} | Tone: {s['tone']} | Format: {s['format']} | Written as: {s['persona']}"
        for i, s in enumerate(specs)
    )

    return f"""Write {TWEETS_PER_API_CALL} short social media posts (tweet-length, under 280 characters each) about US politics.
Each post must follow ITS OWN spec below -- do not blend or repeat structure across posts. The "Written as" field describes the kind of person/account posting -- let it meaningfully shape vocabulary, formality, and structure, not just topic choice:

{spec_lines}

Important instructions for realism and variety:
- Do NOT use a uniform template. Real people don't all write in the "complaint + emoji + hashtag call-to-action" style -- vary sentence structure, length, and opening style across all {TWEETS_PER_API_CALL} posts.
- Not every post needs emoji or a hashtag. Include some posts with NO emoji and NO hashtags at all.
- Vary punctuation and polish: include some posts that are rougher, more fragment-like, or less grammatically perfect, the way real casual tweets are -- not every post should read like polished ad copy.
- For any post using the "reply" format, invent a plausible-looking fake handle (e.g. something like @mike_t2020 or @patriot_jenny) -- never use the literal placeholder text "@someuser".
- Avoid starting multiple posts with the same rhetorical pattern (e.g. don't have several posts all start with a question, or all start with "Just when...").

Return ONLY a JSON array of {TWEETS_PER_API_CALL} strings, in the same order as the specs above, nothing else -- no preamble, no markdown code fences.
Example format: ["post one text here", "post two text here", ...]"""


def generate_batch() -> list[str]:
    prompt = build_prompt()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        temperature=1.1,  # slightly above default -- more lexical variety
        frequency_penalty=0.6,  # discourage repeating the same words/phrases
        presence_penalty=0.4,  # discourage repeating the same topics/structures
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
