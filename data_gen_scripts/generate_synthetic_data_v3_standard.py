"""
Generate diverse synthetic political social-media posts.

Output:
synthetic_tweets.json -> [
  {"text": "...", "label": "ai"},
  ...
]

Includes deliberate casual-style coverage:
- standard posts
- casual lowercase posts
- internet reply-style posts
- terse reactions/fragments
- occasional emoji reactions

This is intended for stylometric classifier experimentation. Synthetic examples
should be paired with comparable real human political posts.
"""

import json
import random
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

client = OpenAI(
    timeout=90.0,
    max_retries=2,
)

MODEL = "gpt-4o-mini"

OUTPUT_FILE = Path("synthetic_tweets_2500_standard.json")
FAILED_BATCH_LOG = Path("failed_batches.log")

NUM_TWEETS_TO_GENERATE = 2500
TWEETS_PER_API_CALL = 10
MAX_BATCH_RETRIES = 3
MAX_TOTAL_BATCH_ATTEMPTS = 850
FINAL_RECOVERY_ATTEMPTS = 250

PROGRESS_STEP = 250
MAX_QUESTION_RATE = 0.18

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# Dataset dimensions
# ---------------------------------------------------------------------

LENGTHS = {
    "very_short": (10, 55),
    "short": (56, 100),
    "medium": (101, 180),
    "long": (181, 270),
}

LENGTH_PROPORTIONS = {
    "very_short": 0.20,
    "short": 0.35,
    "medium": 0.30,
    "long": 0.15,
}

TOPICS = [
    "housing costs and zoning",
    "health insurance and prescription drug prices",
    "student debt and higher education",
    "public schools and school funding",
    "child care and family policy",
    "minimum wage and workers' rights",
    "inflation, groceries, and household costs",
    "tax policy and government spending",
    "immigration and border policy",
    "criminal justice and policing",
    "voting access and election administration",
    "campaign finance and lobbying",
    "climate policy and energy prices",
    "electric vehicles and transportation",
    "trade, tariffs, and domestic manufacturing",
    "reproductive rights",
    "LGBTQ rights",
    "gun policy and public safety",
    "privacy, technology, and social media regulation",
    "foreign policy and military aid",
    "the Supreme Court and federal courts",
    "local infrastructure and public transit",
]

STANCES = [
    "supportive of a progressive policy approach",
    "supportive of a conservative policy approach",
    "skeptical of government intervention",
    "centrist or tradeoff-focused",
    "critical of both major parties",
    "uncertain or still figuring out the issue",
    "focused on practical local effects",
    "not clearly ideological",
]

TONES = [
    "plainspoken",
    "annoyed",
    "curious",
    "dryly funny",
    "hopeful",
    "skeptical",
    "matter-of-fact",
    "tired but restrained",
    "reflective",
    "enthusiastic",
]

FORM_WEIGHTS = {
    "direct reaction": 0.22,
    "personal observation": 0.18,
    "brief opinion": 0.18,
    "question": 0.10,
    "reply-like disagreement": 0.08,
    "dry joke or sarcasm": 0.03,
    "sentence fragment": 0.07,
    "local practical complaint": 0.08,
    "brief supportive reaction": 0.06,
}

# These are intentionally weighted. "standard" remains the majority,
# but the dataset will contain enough casual examples to train against
# casual AI output such as lowercase, slang, short replies, and emoji use.
CASUAL_STYLE_WEIGHTS = {
    "standard": 0.75,
    "casual_lowercase": 0.10,
    "internet_reply": 0.05,
    "short_reaction": 0.08,
    "emoji_reaction": 0.02,
}

# ---------------------------------------------------------------------
# Lightweight quality filtering
# ---------------------------------------------------------------------

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}")
WHITESPACE_RE = re.compile(r"\s+")

METADATA_PATTERNS = [
    r"\b\d{1,3}\s*(?:characters|chars)\b",
    r"\bcharacter count\b",
    r"\bword count\b",
    r"\btweet \d+\b",
    r"\bpost \d+\b",
    r"\btarget length\b",
]

# Intentionally small: catches only recurring generator leakage/cliches.
BOILERPLATE_PATTERNS = [
    r"\bas an ai\b",
    r"\blanguage model\b",
    r"\bit'?s worth noting\b",
    r"\bwhat do you think\b",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def normalize(text: str) -> str:
    text = (
        text.lower()
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
    )
    return WHITESPACE_RE.sub(" ", text).strip()


def word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", normalize(text)))


def opening_key(text: str, word_count: int = 3) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return " ".join(words[:word_count])


def weighted_choice(weight_map: dict[str, float]) -> str:
    return random.choices(
        list(weight_map.keys()),
        weights=list(weight_map.values()),
        k=1,
    )[0]


def even_targets(items: list[str], total: int) -> dict[str, int]:
    base, remainder = divmod(total, len(items))
    result = {item: base for item in items}

    for item in random.sample(items, remainder):
        result[item] += 1

    return result


def proportional_length_targets(total: int) -> dict[str, int]:
    targets = {
        name: int(total * proportion) for name, proportion in LENGTH_PROPORTIONS.items()
    }

    remainder = total - sum(targets.values())

    for name in sorted(
        LENGTH_PROPORTIONS,
        key=LENGTH_PROPORTIONS.get,
        reverse=True,
    ):
        if remainder <= 0:
            break

        targets[name] += 1
        remainder -= 1

    return targets


def too_similar(candidate: str, accepted_texts: list[str]) -> bool:
    """Reject exact duplicates and strong lexical near-duplicates."""
    candidate_normalized = normalize(candidate)
    candidate_words = word_set(candidate)

    for existing in accepted_texts:
        existing_normalized = normalize(existing)

        if candidate_normalized == existing_normalized:
            return True

        character_ratio = SequenceMatcher(
            None,
            candidate_normalized,
            existing_normalized,
        ).ratio()

        if character_ratio >= 0.92:
            return True

        existing_words = word_set(existing)
        all_words = candidate_words | existing_words

        if all_words:
            jaccard = len(candidate_words & existing_words) / len(all_words)

            if jaccard >= 0.87:
                return True

    return False


def repeated_opening(candidate: str, accepted_texts: list[str]) -> bool:
    """
    Allow the same three-word opening twice; reject a third occurrence.
    """
    candidate_opening = opening_key(candidate)

    if not candidate_opening:
        return False

    use_count = sum(
        opening_key(existing) == candidate_opening for existing in accepted_texts
    )

    return use_count >= 5


# ---------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------


def make_profiles(
    accepted_profiles: list[dict],
    batch_size: int,
    length_targets: dict[str, int],
    topic_targets: dict[str, int],
    accepted_question_count: int,
    max_questions: int,
) -> list[dict]:
    """
    Build profiles focused on still-underrepresented topics and length groups.
    """
    length_counts = Counter(profile["length_name"] for profile in accepted_profiles)

    topic_counts = Counter(profile["topic"] for profile in accepted_profiles)

    length_remaining = {
        name: target - length_counts[name] for name, target in length_targets.items()
    }

    topic_remaining = {
        topic: target - topic_counts[topic] for topic, target in topic_targets.items()
    }

    profiles = []

    for _ in range(batch_size):
        available_lengths = [
            name for name, remaining in length_remaining.items() if remaining > 0
        ]

        available_topics = [
            topic for topic, remaining in topic_remaining.items() if remaining > 0
        ]

        if not available_lengths or not available_topics:
            break

        max_length_gap = max(length_remaining[name] for name in available_lengths)

        length_name = random.choice(
            [
                name
                for name in available_lengths
                if length_remaining[name] == max_length_gap
            ]
        )

        max_topic_gap = max(topic_remaining[topic] for topic in available_topics)

        topic = random.choice(
            [
                topic
                for topic in available_topics
                if topic_remaining[topic] == max_topic_gap
            ]
        )

        form = weighted_choice(FORM_WEIGHTS)
        casual_style = weighted_choice(CASUAL_STYLE_WEIGHTS)

        # Short reactions should mostly stay short.
        if casual_style == "short_reaction" and length_name in {"medium", "long"}:
            casual_style = random.choice(
                [
                    "standard",
                    "casual_lowercase",
                    "internet_reply",
                ]
            )

        allow_question = form == "question" and accepted_question_count < max_questions

        if form == "question" and not allow_question:
            form = random.choice(
                [
                    "direct reaction",
                    "personal observation",
                    "brief opinion",
                    "reply-like disagreement",
                    "sentence fragment",
                    "local practical complaint",
                ]
            )

        min_chars, max_chars = LENGTHS[length_name]

        profiles.append(
            {
                "topic": topic,
                "stance": random.choice(STANCES),
                "tone": random.choice(TONES),
                "form": form,
                "casual_style": casual_style,
                "length_name": length_name,
                "min_chars": min_chars,
                "max_chars": max_chars,
                "allow_question": allow_question,
            }
        )

        length_remaining[length_name] -= 1
        topic_remaining[topic] -= 1

    return profiles


# ---------------------------------------------------------------------
# API generation
# ---------------------------------------------------------------------


def build_prompt(profiles: list[dict]) -> str:
    profile_lines = []

    for index, profile in enumerate(profiles, start=1):
        profile_lines.append(
            f"{index}. Topic: {profile['topic']}\n"
            f"   Perspective: {profile['stance']}\n"
            f"   Tone: {profile['tone']}\n"
            f"   Format: {profile['form']}\n"
            f"   Casual style: {profile['casual_style']}\n"
            f"   Length: {profile['min_chars']}-{profile['max_chars']} characters\n"
            f"   Questions allowed: "
            f"{'yes' if profile['allow_question'] else 'no'}"
        )

    return f"""
Generate exactly {len(profiles)} distinct plausible US-political
social-media posts.

Return ONLY valid JSON in exactly this format:

{{"posts": ["post one", "post two", "post three"]}}

Return exactly {len(profiles)} posts in the same order as the profiles.

General rules:
- Each post must be standalone plain text.
- Do not number posts or wrap them in quotation marks.
- Do not mention AI, models, prompts, datasets, character counts, or output
  instructions.
- Do not use URLs or @handles.
- Avoid named politicians and direct quotations.
- Do not fabricate precise statistics, named legislation, named court cases,
  or direct quotes.
- Vary wording and sentence structure.
- Do not make every post a slogan, call to action, polished argument, or
  rhetorical question.
- Follow the assigned character range closely.
- If questions are not allowed, do not use a question mark.

Casual style guide:
- "standard": Natural ordinary political social-media writing.
- "casual_lowercase": Lowercase, contractions, imperfect punctuation, and
  informal wording are okay when natural.
- "internet_reply": Write like a quick reply/reaction. Internet-native
  phrasing such as "idk", "tbh", "rn", "gonna", "bro", "not me", or
  "cool cool" may be used sparingly if it fits. Do not force slang.
- "short_reaction": A terse reaction, fragment, blunt opinion, or
  low-information post is acceptable.
- "emoji_reaction": One emoji may be used naturally. Do not use a hashtag.

Profiles:
{chr(10).join(profile_lines)}
""".strip()


def request_batch(profiles: list[dict]) -> list[str]:
    response = client.chat.completions.create(
        model=MODEL,
        temperature=1.05,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate diverse naturalistic social-media text and "
                    "follow the requested JSON schema exactly."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(profiles),
            },
        ],
    )

    raw_text = response.choices[0].message.content or ""
    parsed = json.loads(raw_text)

    posts = parsed.get("posts")

    if not isinstance(posts, list):
        raise ValueError(
            "Invalid response: missing/non-list 'posts'. "
            f"Raw response: {raw_text[:500]!r}"
        )

    if len(posts) != len(profiles):
        raise ValueError(
            f"Invalid response: expected {len(profiles)} posts, "
            f"got {len(posts)}. Raw response: {raw_text[:500]!r}"
        )

    return posts


# ---------------------------------------------------------------------
# Lightweight validation
# ---------------------------------------------------------------------


def valid_post(
    text: object,
    profile: dict,
    accepted_texts: list[str],
) -> tuple[bool, str]:
    if not isinstance(text, str):
        return False, "not_a_string"

    text = WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return False, "empty"

    if not profile["min_chars"] <= len(text) <= profile["max_chars"]:
        return False, "profile_length_mismatch"

    if URL_RE.search(text):
        return False, "contains_url"

    if HANDLE_RE.search(text):
        return False, "contains_handle"

    normalized = normalize(text)

    if any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in METADATA_PATTERNS
    ):
        return False, "metadata_leakage"

    if any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in BOILERPLATE_PATTERNS
    ):
        return False, "boilerplate"

    if too_similar(text, accepted_texts):
        return False, "near_duplicate"

    if repeated_opening(text, accepted_texts):
        return False, "repeated_opening"

    return True, "ok"


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def actual_length_group(text: str) -> str:
    length = len(text)

    if length <= 55:
        return "very_short"
    if length <= 100:
        return "short"
    if length <= 180:
        return "medium"
    return "long"


def print_distribution(title: str, values: list[str]) -> None:
    print(f"\n{title}:")

    counts = Counter(values)

    for value, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {count:>4}  {value}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    target_lengths = proportional_length_targets(NUM_TWEETS_TO_GENERATE)

    target_topics = even_targets(
        TOPICS,
        NUM_TWEETS_TO_GENERATE,
    )

    max_questions = max(
        1,
        round(NUM_TWEETS_TO_GENERATE * MAX_QUESTION_RATE),
    )

    accepted = []
    accepted_profiles = []
    accepted_texts = []

    rejection_counts = Counter()
    api_failures = 0
    batch_attempts = 0
    next_progress = PROGRESS_STEP

    print(f"Generating {NUM_TWEETS_TO_GENERATE} posts", flush=True)
    print(f"Length targets: {target_lengths}", flush=True)
    print(f"Question limit: {max_questions}", flush=True)

    while (
        len(accepted) < NUM_TWEETS_TO_GENERATE
        and batch_attempts < MAX_TOTAL_BATCH_ATTEMPTS
    ):
        batch_attempts += 1

        remaining = NUM_TWEETS_TO_GENERATE - len(accepted)
        question_count = sum("?" in text for text in accepted_texts)

        profiles = make_profiles(
            accepted_profiles=accepted_profiles,
            batch_size=min(TWEETS_PER_API_CALL, remaining + 2),
            length_targets=target_lengths,
            topic_targets=target_topics,
            accepted_question_count=question_count,
            max_questions=max_questions,
        )

        if not profiles:
            break

        posts = None

        for retry in range(MAX_BATCH_RETRIES):
            try:
                posts = request_batch(profiles)
                break

            except Exception as exc:
                api_failures += 1
                wait_seconds = 1.5 * (retry + 1)

                print(
                    f"Batch failed ({type(exc).__name__}: {exc}); "
                    f"retrying in {wait_seconds:.1f}s",
                    flush=True,
                )

                with FAILED_BATCH_LOG.open("a", encoding="utf-8") as log:
                    log.write("=" * 76 + "\n")
                    log.write(
                        f"Batch {batch_attempts}, "
                        f"retry {retry + 1}/{MAX_BATCH_RETRIES}\n"
                    )
                    log.write(f"{type(exc).__name__}: {exc}\n")
                    log.write(json.dumps(profiles, indent=2))
                    log.write("\n")

                if retry < MAX_BATCH_RETRIES - 1:
                    time.sleep(wait_seconds)

        if posts is None:
            continue

        for post, profile in zip(posts, profiles):
            if len(accepted) >= NUM_TWEETS_TO_GENERATE:
                break

            valid, reason = valid_post(
                text=post,
                profile=profile,
                accepted_texts=accepted_texts,
            )

            if not valid:
                rejection_counts[reason] += 1
                continue

            current_length_counts = Counter(
                item["length_name"] for item in accepted_profiles
            )

            current_topic_counts = Counter(item["topic"] for item in accepted_profiles)

            question_count = sum("?" in text for text in accepted_texts)

            if (
                current_length_counts[profile["length_name"]]
                >= target_lengths[profile["length_name"]]
            ):
                rejection_counts["length_quota_full"] += 1
                continue

            if (
                current_topic_counts[profile["topic"]]
                >= target_topics[profile["topic"]]
            ):
                rejection_counts["topic_quota_full"] += 1
                continue

            if "?" in post and question_count >= max_questions:
                rejection_counts["question_quota_full"] += 1
                continue

            clean_post = WHITESPACE_RE.sub(" ", post).strip()

            accepted.append(
                {
                    "text": clean_post,
                    "label": "ai",
                }
            )

            accepted_profiles.append(profile)
            accepted_texts.append(clean_post)

            while (
                len(accepted) >= next_progress
                and next_progress <= NUM_TWEETS_TO_GENERATE
            ):
                print(
                    f"Progress: {next_progress}/"
                    f"{NUM_TWEETS_TO_GENERATE} tweets completed",
                    flush=True,
                )
                next_progress += PROGRESS_STEP

        time.sleep(0.25)

    # Recovery for final missing quota(s), e.g. 499/500.
    if len(accepted) < NUM_TWEETS_TO_GENERATE:
        print(
            f"Entering recovery at {len(accepted)}/" f"{NUM_TWEETS_TO_GENERATE}",
            flush=True,
        )

        for attempt in range(1, FINAL_RECOVERY_ATTEMPTS + 1):
            if len(accepted) >= NUM_TWEETS_TO_GENERATE:
                break

            question_count = sum("?" in text for text in accepted_texts)

            profiles = make_profiles(
                accepted_profiles=accepted_profiles,
                batch_size=TWEETS_PER_API_CALL,
                length_targets=target_lengths,
                topic_targets=target_topics,
                accepted_question_count=question_count,
                max_questions=max_questions,
            )

            if not profiles:
                break

            print(
                f"Recovery attempt {attempt}/" f"{FINAL_RECOVERY_ATTEMPTS}",
                flush=True,
            )

            try:
                posts = request_batch(profiles)
            except Exception as exc:
                api_failures += 1
                rejection_counts["recovery_api_failure"] += 1

                print(
                    f"Recovery failed ({type(exc).__name__}: {exc})",
                    flush=True,
                )
                time.sleep(1.0)
                continue

            for post, profile in zip(posts, profiles):
                if len(accepted) >= NUM_TWEETS_TO_GENERATE:
                    break

                valid, reason = valid_post(
                    text=post,
                    profile=profile,
                    accepted_texts=accepted_texts,
                )

                if not valid:
                    rejection_counts[f"recovery:{reason}"] += 1
                    continue

                current_length_counts = Counter(
                    item["length_name"] for item in accepted_profiles
                )

                current_topic_counts = Counter(
                    item["topic"] for item in accepted_profiles
                )

                question_count = sum("?" in text for text in accepted_texts)

                if (
                    current_length_counts[profile["length_name"]]
                    >= target_lengths[profile["length_name"]]
                ):
                    continue

                if (
                    current_topic_counts[profile["topic"]]
                    >= target_topics[profile["topic"]]
                ):
                    continue

                if "?" in post and question_count >= max_questions:
                    continue

                clean_post = WHITESPACE_RE.sub(" ", post).strip()

                accepted.append(
                    {
                        "text": clean_post,
                        "label": "ai",
                    }
                )

                accepted_profiles.append(profile)
                accepted_texts.append(clean_post)

                print(
                    f"Recovery accepted: {len(accepted)}/" f"{NUM_TWEETS_TO_GENERATE}",
                    flush=True,
                )

                while (
                    len(accepted) >= next_progress
                    and next_progress <= NUM_TWEETS_TO_GENERATE
                ):
                    print(
                        f"Progress: {next_progress}/"
                        f"{NUM_TWEETS_TO_GENERATE} tweets completed",
                        flush=True,
                    )
                    next_progress += PROGRESS_STEP

            time.sleep(0.25)

    if len(accepted) < NUM_TWEETS_TO_GENERATE:
        length_counts = Counter(profile["length_name"] for profile in accepted_profiles)

        raise RuntimeError(
            f"Only generated {len(accepted)}/"
            f"{NUM_TWEETS_TO_GENERATE} posts. "
            f"Length counts: {dict(length_counts)}"
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(accepted, file, indent=2, ensure_ascii=False)

    texts = [item["text"] for item in accepted]
    text_lengths = sorted(len(text) for text in texts)

    print("\n" + "=" * 72)
    print(f"Wrote {len(accepted)} synthetic AI posts to {OUTPUT_FILE}")
    print(f"API/batch failures after retries: {api_failures}")
    print(f"Rejected outputs: {sum(rejection_counts.values())}")
    print(f"Rejection reasons: {dict(rejection_counts)}")

    print(
        f"Length: min={min(text_lengths)}, "
        f"median={text_lengths[len(text_lengths) // 2]}, "
        f"max={max(text_lengths)}"
    )

    print(f"Questions: {sum('?' in text for text in texts)}/{len(texts)}")

    print(f"Hashtags: {sum('#' in text for text in texts)}/{len(texts)}")

    print(
        f"Likely emojis: "
        f"{sum(any(ord(ch) > 0x1F000 for ch in text) for text in texts)}/"
        f"{len(texts)}"
    )

    print_distribution(
        "Target length groups",
        [profile["length_name"] for profile in accepted_profiles],
    )

    print_distribution(
        "Actual length groups",
        [actual_length_group(text) for text in texts],
    )

    print_distribution(
        "Topic coverage",
        [profile["topic"] for profile in accepted_profiles],
    )

    print_distribution(
        "Casual-style coverage",
        [profile["casual_style"] for profile in accepted_profiles],
    )

    print_distribution(
        "Post-form coverage",
        [profile["form"] for profile in accepted_profiles],
    )

    opening_counts = Counter(opening_key(text) for text in texts)
    repeated_openings = {
        opening: count
        for opening, count in opening_counts.items()
        if opening and count >= 2
    }

    print("\nRepeated three-word openings:")
    print(repeated_openings if repeated_openings else "None")
    print("=" * 72)


if __name__ == "__main__":
    main()
