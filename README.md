# AI Text & Sentiment Detector (Browser Extension — Python Prototyping Phase)

A browser extension that predicts (1) the likelihood a piece of text is
AI-generated, and (2) where it falls on a scale from strongly negative to
strongly positive. This repo currently holds the Python model prototyping
phase, before the extension itself is built.

## Status

- [x] Sentiment scale prototype (single pretrained model, continuous score)
- [x] AI-likelihood detector prototype (pretrained HC3-based classifier — placeholder, poor fit for short political posts, see Known limitations)
- [x] Local Flask test server with textbox UI showing both scores side by side
- [x] Synthetic AI-generated political tweet dataset (to pair with real human tweets already on hand)
- [x] Custom fine-tuned AI-detection model, trained on political-post-domain data
- [x] Browser extension (manifest, content script, in-browser inference)

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

at minimum you'll need:

```bash
pip install torch transformers scikit-learn pandas langdetect flask datasets accelerate
```

## Files

## Files

| File                                           | Purpose                                                                                                                                                                                                                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sentiment.py`                                 | Core sentiment scoring logic. Loads the pretrained 3-class sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) and derives a continuous score (`P(positive) - P(negative)`, range -1 to +1) plus a human-readable label (e.g. `"strongly positive"`). |
| `detector.py`                                  | AI-generated text detection. Currently uses a pretrained HC3-based classifier (`Hello-SimpleAI/chatgpt-detector-roberta`) as a placeholder — see **Known Limitations**. `detect(text)` is the swap point for the custom model later.                                   |
| `sample.py`                                    | Loads a random sample of tweets from a local dataset CSV, cleans and language-filters them, and runs them through `analyze()` as a stress test on real, messy text.                                                                                                    |
| `app.py`                                       | Local Flask server with a textbox UI for interactively testing arbitrary text against both the sentiment scale and AI-likelihood detector. Run and open `http://127.0.0.1:5000`.                                                                                       |
| `data_gen_scripts/generate_synthetic_data*.py` | Uses OpenAI to generate a JSON file of AI-generated political tweets for training data.                                                                                                                                                                                |
| `extension/`                                   | Packages the custom predictor into a working browser extension. Contains the frontend and backend logic. Requires `~/app.py` to be running.                                                                                                                            |
| `train_custom_predictor.py`                    | Fine-tunes a RoBERTa classifier for AI-text detection using a self-generated training dataset.                                                                                                                                                                         |

## Dataset (sentiment / real human tweets)

`sample.py` expects the **US Election 2020 Tweets** dataset from Kaggle:

https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets

Download `hashtag_donaldtrump.csv` (and/or `hashtag_joebiden.csv`) and place
it in the project root. These files are large and are **not** committed to
this repo (see `.gitignore`) — download them yourself from the link above.

Update `CSV_PATH` at the top of `sample.py` if your filename differs.

This dataset also doubles as the "human" class for the AI-detection model —
see Next steps.

## Running things

Download our trained model from [Google Drive](https://drive.google.com/drive/folders/11E4_eX1LTobrdhfyAHmNSj078-VVYplg?usp=drive_link) OR
train your own by generating tweets by calling the scripts in data_gen_scripts/ and then
running train_custom_predictor.py

```bash
# Launch the interactive test server (sentiment + AI likelihood side by side)
python app.py
# then open http://127.0.0.1:5000
```

Open a new chrome tab. Navigate to manage extensions. Enable developer mode (toggle from the top right), press the
'Load unpacked' button and select the extension/ folder. Enable the custom extension and you are good to go!.

For more testing:

```bash
# Sanity-check the sentiment scale on a few hardcoded examples
python sentiment.py

# Sanity-check the AI detector on a few hardcoded examples
python detector.py

# Stress-test sentiment against a sample of real tweets from the dataset
python sample.py
```

## Known limitations

- The sentiment model is English-only; non-English text is filtered out in
  `sample.py` via `langdetect` rather than scored (it would produce
  unreliable results if scored directly).
- No sarcasm detection — sarcastic text is a known hard failure mode for
  sentiment models generally, not something fixed here.
- The bucket thresholds in `label_for_score()` (e.g. what counts as
  "strongly" positive/negative) are a rough starting guess, not derived
  from labeled data yet.
- **The current AI detector (`detector.py`) is a known poor fit** for this
  project's actual use case (short political posts). It's trained on
  long-form Q&A text (HC3 dataset) and in testing confidently mislabeled
  real, coherent human political tweets as AI-generated (~99.9% confidence)
  while correctly scoring choppy/hashtag-heavy tweets as human — it appears
  to be picking up on formality/grammatical completeness rather than actual
  AI-authorship signal. Treat its output as a placeholder for wiring up the
  demo UI, not a meaningful signal yet.
- The TweepFake dataset (`mtesconi/twitter-deep-fake-text` on Kaggle) was
  evaluated as a fine-tuning source and rejected: (1) its bot tweets were
  generated with older techniques (Markov chains, RNN, GPT-2, collected in
  2020), not modern LLMs, and (2) the dataset only ships tweet IDs, not
  text — hydrating via the X API costs real money (~$0.005/read, no free
  tier as of 2026) and many of the original bot accounts are likely
  suspended by now, meaning an incomplete dataset even after paying.

## Next steps

Refine the custom dataset:

- **Human class:** sample directly from the already-downloaded election
  tweets CSVs (real text, no cost, already domain-matched).
- **AI class:** Improve the prompting to create better synthetic tweets.
  If possible sample from more models besides openAI.
