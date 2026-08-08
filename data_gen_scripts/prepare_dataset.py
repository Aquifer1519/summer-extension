"""
Build a balanced human-vs-AI political-tweet training dataset.

AI training inputs:
    - synthetic_tweets_2500_standard.json
    - synthetic_tweets_2500_casual.json
    - synthetic_tweets_2500_commentary.json

Important:
    Do NOT include synthetic_tweets_2500_holdout.json here.
    Keep it for later external/generalization testing.

Outputs:
    data/synthetic_tweets_7500_train.json
    data/train.csv
    data/val.csv
    data/test.csv

Each CSV has:
    text,label

where label is "human" or "ai".
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# Folder containing this script: .../data_gen_scripts/
SCRIPT_DIR = Path(__file__).resolve().parent

# Parent project folder: .../summer-extension/
PROJECT_DIR = SCRIPT_DIR.parent

# Output folder: .../summer-extension/data/
DATA_DIR = PROJECT_DIR / "data"

# Allow importing sample.py if it is in the project root.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from sample import clean_tweet, is_english

# ---------------------------------------------------------------------
# Input / output files
# ---------------------------------------------------------------------

# AI JSON files are stored alongside this script in data_gen_scripts/.
AI_INPUT_FILES = [
    SCRIPT_DIR / "synthetic_tweets_2500_standard.json",
    SCRIPT_DIR / "synthetic_tweets_2500_casual.json",
    SCRIPT_DIR / "synthetic_tweets_2500_commentary.json",
]

# Kept separate from training.
AI_HOLDOUT_FILE = SCRIPT_DIR / "synthetic_tweets_2500_holdout.json"

# Human source files are in ../data/.
HUMAN_CSV_FILES = [
    DATA_DIR / "hashtag_donaldtrump.csv",
    DATA_DIR / "hashtag_joebiden.csv",
]

HUMAN_TEXT_COLUMN = "tweet"

# All generated training files go to ../data/.
COMBINED_AI_OUT = DATA_DIR / "synthetic_tweets_7500_train.json"
TRAIN_OUT = DATA_DIR / "train.csv"
VAL_OUT = DATA_DIR / "val.csv"
TEST_OUT = DATA_DIR / "test.csv"

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# AI loading / combining
# ---------------------------------------------------------------------


def load_ai_file(json_path: Path) -> list[dict]:
    """Load, clean, English-filter, and label one synthetic JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(
            f"AI input file not found: {json_path.resolve()}\n"
            "Check AI_INPUT_FILES in prepare_dataset.py."
        )

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in {json_path}, got {type(data).__name__}."
        )

    cleaned_items = []

    for item in data:
        if not isinstance(item, dict):
            continue

        raw_text = str(item.get("text", "")).strip()

        # Apply the same text-cleaning logic used for human examples.
        # This prevents URLs/mentions/etc. from becoming a class shortcut.
        cleaned_text = clean_tweet(raw_text)

        if cleaned_text and is_english(cleaned_text):
            cleaned_items.append(
                {
                    "text": cleaned_text,
                    "label": "ai",
                    "source_file": json_path.name,
                }
            )

    print(f"Loaded {len(cleaned_items)} usable AI tweets from " f"{json_path.name}")

    return cleaned_items


def load_and_combine_ai_tweets() -> list[dict]:
    """
    Combine all configured AI training files and deduplicate by normalized text.

    The source_file field is retained temporarily for diagnostics, then removed
    before writing train/val/test CSVs.
    """
    combined = []

    print("\nLoading AI training files:")

    for json_path in AI_INPUT_FILES:
        combined.extend(load_ai_file(json_path))

    if not combined:
        raise RuntimeError(
            "No usable AI tweets were loaded. Check AI_INPUT_FILES and "
            "the input JSON format."
        )

    ai_df = pd.DataFrame(combined)

    print(f"\nAI tweets before deduplication: {len(ai_df)}")

    # Case-insensitive normalized text key for cross-file deduplication.
    ai_df["dedup_key"] = (
        ai_df["text"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    ai_df = ai_df.drop_duplicates(subset="dedup_key").copy()

    print(f"AI tweets after deduplication:  {len(ai_df)}")

    print("\nAI source distribution after deduplication:")
    print(ai_df["source_file"].value_counts())

    ai_df = ai_df.drop(columns=["dedup_key"])

    # Save a combined AI-only training file for reproducibility.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    combined_ai_for_json = ai_df[["text", "label"]].to_dict(orient="records")

    with COMBINED_AI_OUT.open("w", encoding="utf-8") as file:
        json.dump(
            combined_ai_for_json,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nWrote combined AI file: {COMBINED_AI_OUT}")
    print(f"AI holdout file intentionally excluded: {AI_HOLDOUT_FILE}")

    return ai_df.to_dict(orient="records")


# ---------------------------------------------------------------------
# Human tweet sampling
# ---------------------------------------------------------------------


def collect_human_candidates_from_file(
    csv_path: Path,
    target_candidates: int,
) -> list[dict]:
    """
    Collect cleaned English human tweets from one source CSV.

    target_candidates is larger than the final desired amount because
    duplicates and invalid rows are removed later.
    """
    if not csv_path.exists():
        print(f"Warning: human CSV not found, skipping: {csv_path}")
        return []

    collected = []

    try:
        chunk_iter = pd.read_csv(
            csv_path,
            engine="python",
            on_bad_lines="skip",
            encoding="utf-8",
            chunksize=5000,
            usecols=[HUMAN_TEXT_COLUMN],
        )

        for chunk in chunk_iter:
            texts = chunk[HUMAN_TEXT_COLUMN].dropna().astype(str).str.strip()

            texts = texts[texts.str.len() > 0]

            for raw_text in texts:
                cleaned_text = clean_tweet(raw_text)

                if cleaned_text and is_english(cleaned_text):
                    collected.append(
                        {
                            "text": cleaned_text,
                            "label": "human",
                            "source_file": csv_path.name,
                        }
                    )

            if len(collected) >= target_candidates:
                break

    except pd.errors.ParserError as error:
        print(
            f"Warning: parser error in {csv_path}. "
            f"Keeping {len(collected)} rows collected before the error.\n"
            f"Details: {error}"
        )

    print(f"Collected {len(collected)} human candidates from " f"{csv_path.name}")

    return collected


def sample_human_tweets(n: int) -> list[dict]:
    """
    Collect a roughly balanced sample across all configured human source CSVs.

    If one source has too few usable tweets, the final sample can still use
    additional tweets from the other source.
    """
    if not HUMAN_CSV_FILES:
        raise RuntimeError("HUMAN_CSV_FILES is empty.")

    print("\nCollecting human tweets:")

    # Ask each source for more than its final share to leave room for dedup.
    per_source_target = int((n / len(HUMAN_CSV_FILES)) * 1.7) + 100

    all_candidates = []

    for csv_path in HUMAN_CSV_FILES:
        all_candidates.extend(
            collect_human_candidates_from_file(
                csv_path=csv_path,
                target_candidates=per_source_target,
            )
        )

    if not all_candidates:
        raise RuntimeError(
            "No usable human tweets were collected. Check HUMAN_CSV_FILES, "
            "HUMAN_TEXT_COLUMN, and sample.py."
        )

    human_df = pd.DataFrame(all_candidates)

    print(f"\nHuman candidates before deduplication: {len(human_df)}")

    human_df["dedup_key"] = (
        human_df["text"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    human_df = human_df.drop_duplicates(subset="dedup_key").copy()

    print(f"Human candidates after deduplication:  {len(human_df)}")

    print("\nHuman source distribution before sampling:")
    print(human_df["source_file"].value_counts())

    if len(human_df) < n:
        print(
            f"\nWARNING: only {len(human_df)} usable human tweets were "
            f"found, fewer than the {n} needed to match the AI class."
        )
        n = len(human_df)

    # Sample as evenly as possible from each source where possible.
    source_names = list(human_df["source_file"].unique())
    per_source_n = n // len(source_names)
    remainder = n % len(source_names)

    samples = []

    for index, source_name in enumerate(source_names):
        source_df = human_df[human_df["source_file"] == source_name]

        requested = per_source_n + (1 if index < remainder else 0)
        take = min(requested, len(source_df))

        if take > 0:
            samples.append(
                source_df.sample(
                    n=take,
                    random_state=RANDOM_SEED + index,
                )
            )

    sampled_df = pd.concat(samples, ignore_index=True)

    # If one source did not have enough tweets, fill the remaining count
    # from unused examples in the combined human candidate pool.
    if len(sampled_df) < n:
        selected_keys = set(sampled_df["dedup_key"])

        remaining_df = human_df[~human_df["dedup_key"].isin(selected_keys)]

        needed = min(n - len(sampled_df), len(remaining_df))

        if needed > 0:
            fill_df = remaining_df.sample(
                n=needed,
                random_state=RANDOM_SEED,
            )
            sampled_df = pd.concat(
                [sampled_df, fill_df],
                ignore_index=True,
            )

    sampled_df = sampled_df.drop(columns=["dedup_key"])

    print("\nHuman source distribution after sampling:")
    print(sampled_df["source_file"].value_counts())

    return sampled_df.to_dict(orient="records")


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def print_dataset_diagnostics(df: pd.DataFrame) -> None:
    """Print useful checks before splitting the final combined dataset."""
    print(f"\nFinal combined dataset: {len(df)} rows")
    print("\nClass counts:")
    print(df["label"].value_counts())

    print("\nCharacter-length summary by label:")
    print(
        df.assign(char_length=df["text"].str.len())
        .groupby("label")["char_length"]
        .describe()[["count", "mean", "min", "50%", "max"]]
    )

    print("\nExamples after cleaning:")

    for label in ["human", "ai"]:
        print(f"\n{label.upper()} examples:")

        examples = (
            df[df["label"] == label]["text"]
            .sample(
                n=min(5, (df["label"] == label).sum()),
                random_state=RANDOM_SEED,
            )
            .tolist()
        )

        for text in examples:
            print(f"  - {text}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    if abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) > 1e-9:
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must equal 1.0.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("AI input files:")
    for filename in AI_INPUT_FILES:
        print(f"  - {filename.resolve()}")

    ai_tweets = load_and_combine_ai_tweets()
    human_tweets = sample_human_tweets(len(ai_tweets))

    # Keep classes balanced if there were fewer human tweets available.
    final_class_size = min(len(ai_tweets), len(human_tweets))

    if len(ai_tweets) != final_class_size:
        print(
            f"\nDownsampling AI class from {len(ai_tweets)} to "
            f"{final_class_size} to keep classes balanced."
        )

        ai_tweets = random.Random(RANDOM_SEED).sample(
            ai_tweets,
            final_class_size,
        )

    if len(human_tweets) != final_class_size:
        human_tweets = random.Random(RANDOM_SEED).sample(
            human_tweets,
            final_class_size,
        )

    # Remove source-file metadata before training.
    all_data = [
        {
            "text": item["text"],
            "label": item["label"],
        }
        for item in (ai_tweets + human_tweets)
    ]

    random.Random(RANDOM_SEED).shuffle(all_data)

    df = pd.DataFrame(all_data)
    print_dataset_diagnostics(df)

    # 80% train; the remaining 20% is split equally into validation and test.
    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_FRAC,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    val_relative_frac = VAL_FRAC / (VAL_FRAC + TEST_FRAC)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_frac,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print("\nSplit class counts:")

    for name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        print(f"\n{name} ({len(split_df)} rows):")
        print(split_df["label"].value_counts())

    print(
        f"\nWrote:\n"
        f"  {TRAIN_OUT} ({len(train_df)} rows)\n"
        f"  {VAL_OUT} ({len(val_df)} rows)\n"
        f"  {TEST_OUT} ({len(test_df)} rows)"
    )


if __name__ == "__main__":
    main()
