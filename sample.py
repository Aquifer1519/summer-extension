"""
Load a random sample from the US Election 2020 Tweets dataset and run it
through the sentiment/emotion pipeline as a stress test on real, messy text.

Assumes you've downloaded the CSV from Kaggle (e.g. hashtag_donaldtrump.csv
or hashtag_joebiden.csv) and have `sentiment.py` in the same directory.
"""

import re
import pandas as pd
from sentiment import analyze

CSV_PATH = (
    "data\hashtag_donaldtrump.csv"  # hashtag_donaldtrump.csv or hashtag_joebiden.csv
)
TEXT_COLUMN = "tweet"  # update if the column name differs
SAMPLE_SIZE = 25
CANDIDATE_POOL_SIZE = 500  # pull this many candidates first, then filter down
RANDOM_SEED = 42

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_tweet(text: str) -> str:
    """Strip URLs and @mentions -- noise tokens that don't carry sentiment,
    and can distort embeddings the model wasn't trained to ignore."""
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    return " ".join(text.split())  # collapse extra whitespace left behind


from langdetect import detect, LangDetectException


def is_english(text: str) -> bool:
    """Real language detection via langdetect (pip install langdetect).
    Far more reliable than an ASCII-ratio heuristic -- French, Italian,
    Spanish, etc. are mostly ASCII too, so ASCII-ratio alone doesn't
    catch them."""
    if not text or len(text) < 3:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def load_sample(csv_path: str, text_col: str, n: int) -> list[str]:
    # Read in CHUNKS with the default (fast) C engine + on_bad_lines="skip".
    # This avoids two problems at once:
    #   1. engine="python" parsing the whole 1.7M-row file, which is slow
    #      and gives no progress feedback for minutes at a time.
    #   2. Loading the entire file into memory when we only need a few
    #      hundred candidate rows.
    # We stop as soon as we've collected enough candidates -- no need to
    # scan the whole file.
    collected = []
    chunk_iter = pd.read_csv(
        csv_path,
        on_bad_lines="skip",  # C engine supports this directly, no need for engine="python"
        encoding="utf-8",
        chunksize=5000,
        usecols=[text_col],
    )

    for i, chunk in enumerate(chunk_iter):
        print(f"  scanning chunk {i + 1} (~{(i + 1) * 5000} rows so far)...")
        texts = chunk[text_col].dropna().astype(str).str.strip()
        texts = texts[texts.str.len() > 0]
        collected.extend(texts.tolist())

        if len(collected) >= CANDIDATE_POOL_SIZE:
            break

    print(f"Collected {len(collected)} raw candidate rows. Cleaning + filtering...")

    candidates = pd.Series(collected).drop_duplicates()
    cleaned = candidates.map(clean_tweet)
    cleaned = cleaned[cleaned.map(is_english)]
    cleaned = cleaned[cleaned.str.len() > 0]

    if len(cleaned) < n:
        print(
            f"Warning: only {len(cleaned)} English tweets survived filtering "
            f"out of {len(candidates)} candidates. Increase CANDIDATE_POOL_SIZE "
            f"for a full sample of {n}."
        )

    sample = cleaned.sample(n=min(n, len(cleaned)), random_state=RANDOM_SEED)
    return sample.tolist()


if __name__ == "__main__":
    tweets = load_sample(CSV_PATH, TEXT_COLUMN, SAMPLE_SIZE)
    print(f"Loaded {len(tweets)} sample tweets from {CSV_PATH}\n")

    for tweet in tweets:
        result = analyze(tweet)
        print(f"TEXT: {result['text']}")
        print(f"  Scale: {result['scale_score']:+.4f}  ({result['scale_label']})")
        print("-" * 60)
