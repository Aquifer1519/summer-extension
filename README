# AI Text & Sentiment Detector (Browser Extension — Python Prototyping Phase)

A browser extension that predicts (1) the likelihood a piece of text is
AI-generated, and (2) where it falls on a scale from strongly negative to
strongly positive. This repo currently holds the Python model prototyping
phase, before the extension itself is built.

## Status

- [x] Sentiment scale prototype (single pretrained model, continuous score)
- [x] Local Flask test server with a textbox UI
- [ ] AI-generated text detection model
- [ ] Browser extension (manifest, content script, in-browser inference)

## Setup

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, at minimum you'll need:

```bash
pip install torch transformers scikit-learn pandas langdetect flask
```

## Files

| File               | Purpose                                                                                                                                                                                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sentiment.py`     | Core scoring logic. Loads a pretrained 3-class sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and derives a continuous score (`P(positive) - P(negative)`, range -1 to +1) plus a human-readable label (e.g. "strongly positive"). |
| `sample_tweets.py` | Loads a random sample of tweets from a local dataset CSV (see below), cleans and language-filters them, and runs them through `analyze()` as a stress test on real, messy text.                                                                          |
| `app.py`           | Local Flask server with a textbox UI for interactively testing arbitrary text against the sentiment scale. Run and open `http://127.0.0.1:5000`.                                                                                                         |

## Dataset

`sample_tweets.py` expects the **US Election 2020 Tweets** dataset from Kaggle:

https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets

Download `hashtag_donaldtrump.csv` (and/or `hashtag_joebiden.csv`) and place
it in the project root. These files are large and are **not** committed to
this repo (see `.gitignore`) — download them yourself from the link above.

Update `CSV_PATH` at the top of `sample_tweets.py` if your filename differs.

## Running things

```bash
# Sanity-check the sentiment scale on a few hardcoded examples
python sentiment.py

# Stress-test against a sample of real tweets from the dataset
python sample_tweets.py

# Launch the interactive test server
python app.py
# then open http://127.0.0.1:5000
```

## Known limitations

- The sentiment model is English-only; non-English text is filtered out in
  `sample_tweets.py` via `langdetect` rather than scored (it would produce
  unreliable results if scored directly).
- No sarcasm detection — sarcastic text is a known hard failure mode for
  sentiment models generally, not something fixed here.
- The bucket thresholds in `label_for_score()` (e.g. what counts as
  "strongly" positive/negative) are a rough starting guess, not derived
  from labeled data yet.

## Next steps

AI-generated text detection is still unbuilt. Planned approach: fine-tune
a RoBERTa-based classifier on the HC3 (Human ChatGPT Comparison Corpus)
dataset, or evaluate an existing pretrained detector zero-shot before
committing to training one from scratch.
