"""
Combine the synthetic AI-generated tweets (synthetic_tweets.json) with a
matching sample of real human tweets (from the election dataset CSVs) into
a single labeled dataset, split into train/val/test for fine-tuning.

Usage:
    python prepare_dataset.py

Output: train.csv, val.csv, test.csv -- each with columns "text","label"
where label is "human" or "ai". These are what fine_tune_detector.py
expects.
"""

import json
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

# Reuse the same cleaning/filtering logic from sample_tweets.py so the
# human class gets the same treatment (strip URLs/mentions, English-only)
from sample import clean_tweet, is_english

DATA_DIR = "data"
SYNTHETIC_JSON = os.path.join(DATA_DIR, "synthetic_tweets.json")
HUMAN_CSV_FILES = [
    os.path.join(DATA_DIR, "hashtag_donaldtrump.csv"),
    os.path.join(DATA_DIR, "hashtag_joebiden.csv"),
]
HUMAN_TEXT_COLUMN = "tweet"

TRAIN_OUT = os.path.join(DATA_DIR, "train.csv")
VAL_OUT = os.path.join(DATA_DIR, "val.csv")
TEST_OUT = os.path.join(DATA_DIR, "test.csv")

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1  # must sum to 1.0 with the above

RANDOM_SEED = 42


def load_synthetic_ai_tweets() -> list[dict]:
    with open(SYNTHETIC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize label just in case, and drop any accidental empties
    cleaned = [
        {"text": item["text"].strip(), "label": "ai"}
        for item in data
        if item.get("text", "").strip()
    ]
    print(f"Loaded {len(cleaned)} synthetic AI tweets")
    return cleaned


def sample_human_tweets(n: int) -> list[dict]:
    """Pull real human tweets across all provided CSVs, cleaned and
    English-filtered, until we have n (or run out)."""
    collected = []

    for csv_path in HUMAN_CSV_FILES:
        if len(collected) >= n:
            break
        try:
            chunk_iter = pd.read_csv(
                csv_path,
                engine="python",  # more tolerant of malformed rows (unterminated
                # quotes etc.) than the default C engine, which
                # can hard-crash with "Buffer overflow" on rows
                # it can't even find the boundary of -- slower,
                # but we're reading deep enough into the file
                # now that we can't just get lucky and avoid it.
                on_bad_lines="skip",
                encoding="utf-8",
                chunksize=5000,
                usecols=[HUMAN_TEXT_COLUMN],
            )
        except FileNotFoundError:
            print(f"  Warning: {csv_path} not found, skipping.")
            continue

        try:
            for chunk in chunk_iter:
                texts = chunk[HUMAN_TEXT_COLUMN].dropna().astype(str).str.strip()
                texts = texts[texts.str.len() > 0]

                for t in texts:
                    cleaned = clean_tweet(t)
                    if cleaned and is_english(cleaned):
                        collected.append({"text": cleaned, "label": "human"})

                if len(collected) >= n * 1.2:  # small buffer before dedup/sample
                    break
        except pd.errors.ParserError as e:
            print(
                f"  Warning: parser error partway through {csv_path}, "
                f"using what was collected so far ({len(collected)} rows). Error: {e}"
            )

    print(f"Collected {len(collected)} candidate human tweets before dedup")

    df = pd.DataFrame(collected).drop_duplicates(subset="text")
    print(f"  {len(df)} remain after dedup")

    if len(df) < n:
        print(
            f"  WARNING: only found {len(df)} human tweets, fewer than the "
            f"{n} requested to match the AI class. Consider adding more "
            f"CSV files to HUMAN_CSV_FILES, or reducing the AI class size."
        )
        n = len(df)

    sampled = df.sample(n=n, random_state=RANDOM_SEED)
    return sampled.to_dict(orient="records")


def main():
    ai_tweets = load_synthetic_ai_tweets()
    human_tweets = sample_human_tweets(len(ai_tweets))

    all_data = ai_tweets + human_tweets
    random.Random(RANDOM_SEED).shuffle(all_data)

    df = pd.DataFrame(all_data)
    print(f"\nFinal combined dataset: {len(df)} rows")
    print(df["label"].value_counts())

    # First split off train, then split the remainder into val/test
    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_FRAC, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_relative_frac = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_frac,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print(
        f"\nWrote {TRAIN_OUT} ({len(train_df)}), {VAL_OUT} ({len(val_df)}), {TEST_OUT} ({len(test_df)})"
    )


if __name__ == "__main__":
    main()
