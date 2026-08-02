"""
Local test server: sentiment scale + AI-likelihood detector, side by side.

Run with:
    pip install flask
    python app.py

Then open http://127.0.0.1:5000 in your browser. Type or paste text,
hit Analyze (or press Ctrl+Enter), and see both scores update without a
page reload.

NOTE: the AI-likelihood bar currently uses the pretrained HC3-based
detector from detector.py (Hello-SimpleAI/chatgpt-detector-roberta).
As discussed, this is a placeholder -- known to be a poor domain fit for
short political posts (see detector.py's docstring/notes) -- swap in
the custom fine-tuned model once it exists. detect() is imported directly
so that swap only requires changing detector.py, not this file.
"""

from flask import Flask, request, jsonify, render_template_string
from sentiment import analyze
from detector import detect

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Text Analysis Tester</title>
  <style>
    body {
      font-family: -apple-system, Segoe UI, Roboto, sans-serif;
      max-width: 640px;
      margin: 40px auto;
      padding: 0 20px;
      color: #222;
    }
    h1 { font-size: 20px; }
    textarea {
      width: 100%;
      height: 120px;
      font-size: 15px;
      padding: 10px;
      box-sizing: border-box;
      border: 1px solid #ccc;
      border-radius: 6px;
    }
    button {
      margin-top: 10px;
      padding: 8px 16px;
      font-size: 14px;
      cursor: pointer;
      border: none;
      border-radius: 6px;
      background: #222;
      color: white;
    }
    button:disabled { opacity: 0.5; cursor: default; }
    .panel {
      margin-top: 20px;
      padding: 16px;
      border-radius: 8px;
      background: #f5f5f5;
      display: none;
    }
    .panel h2 {
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #888;
      margin: 0 0 8px 0;
    }
    .bar-track {
      position: relative;
      height: 24px;
      border-radius: 12px;
      margin: 12px 0;
    }
    #sentiment-track {
      background: linear-gradient(to right, #d32f2f, #eee 50%, #2e7d32);
    }
    #ai-track {
      background: linear-gradient(to right, #2e7d32, #eee 50%, #d32f2f);
    }
    .bar-marker {
      position: absolute;
      top: -4px;
      width: 4px;
      height: 32px;
      background: #222;
      border-radius: 2px;
      transform: translateX(-2px);
    }
    .label { font-size: 18px; font-weight: 600; }
    .score { color: #555; font-size: 14px; }
    .scale-legend { font-size:12px;color:#888;display:flex;justify-content:space-between; }
    .error { color: #d32f2f; }
    .note { font-size: 11px; color: #aaa; margin-top: 8px; }
  </style>
</head>
<body>
  <h1>Text Analysis Tester</h1>
  <textarea id="text" placeholder="Type or paste text here... (Ctrl+Enter to analyze)"></textarea>
  <br>
  <button id="analyze-btn">Analyze</button>

  <div id="sentiment-panel" class="panel">
    <h2>Sentiment</h2>
    <div id="sentiment-label" class="label"></div>
    <div id="sentiment-score" class="score"></div>
    <div id="sentiment-track" class="bar-track"><div id="sentiment-marker" class="bar-marker"></div></div>
    <div class="scale-legend">
      <span>strongly negative</span><span>neutral</span><span>strongly positive</span>
    </div>
  </div>

  <div id="ai-panel" class="panel">
    <h2>AI Likelihood</h2>
    <div id="ai-label" class="label"></div>
    <div id="ai-score" class="score"></div>
    <div id="ai-track" class="bar-track"><div id="ai-marker" class="bar-marker"></div></div>
    <div class="scale-legend">
      <span>likely human</span><span>uncertain</span><span>likely AI</span>
    </div>
    <div class="note">Using pretrained HC3-based detector (placeholder) -- not tuned for short political posts. Will be swapped for a custom-trained model.</div>
  </div>

  <script>
    const btn = document.getElementById('analyze-btn');
    const textEl = document.getElementById('text');

    const sentimentPanel = document.getElementById('sentiment-panel');
    const sentimentLabel = document.getElementById('sentiment-label');
    const sentimentScore = document.getElementById('sentiment-score');
    const sentimentMarker = document.getElementById('sentiment-marker');

    const aiPanel = document.getElementById('ai-panel');
    const aiLabel = document.getElementById('ai-label');
    const aiScore = document.getElementById('ai-score');
    const aiMarker = document.getElementById('ai-marker');

    function showError(panel, labelEl, scoreEl, message) {
      labelEl.textContent = 'Error';
      labelEl.className = 'label error';
      scoreEl.textContent = message;
      panel.style.display = 'block';
    }

    function aiLabelForScore(score) {
      if (score >= 0.7) return 'likely AI';
      if (score >= 0.4) return 'uncertain';
      return 'likely human';
    }

    async function runAnalysis() {
      const text = textEl.value.trim();
      if (!text) return;

      btn.disabled = true;
      btn.textContent = 'Analyzing...';

      try {
        const res = await fetch('/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text })
        });
        const data = await res.json();

        if (data.error) {
          showError(sentimentPanel, sentimentLabel, sentimentScore, data.error);
          showError(aiPanel, aiLabel, aiScore, data.error);
        } else {
          // --- Sentiment panel ---
          sentimentLabel.textContent = data.sentiment.scale_label;
          sentimentLabel.className = 'label';
          sentimentScore.textContent = `score: ${data.sentiment.scale_score.toFixed(4)}  |  raw: ${JSON.stringify(data.sentiment.raw_scores)}`;
          const sentPct = ((data.sentiment.scale_score + 1) / 2) * 100;  // -1..1 -> 0..100
          sentimentMarker.style.left = sentPct + '%';
          sentimentPanel.style.display = 'block';

          // --- AI likelihood panel ---
          const aiVal = data.ai_detection.ai_likelihood;
          aiLabel.textContent = aiLabelForScore(aiVal);
          aiLabel.className = 'label';
          aiScore.textContent = `AI likelihood: ${aiVal.toFixed(4)}  |  raw: ${JSON.stringify(data.ai_detection.raw_scores)}`;
          const aiPct = aiVal * 100;  // 0..1 -> 0..100
          aiMarker.style.left = aiPct + '%';
          aiPanel.style.display = 'block';
        }
      } catch (err) {
        showError(sentimentPanel, sentimentLabel, sentimentScore, 'Could not reach the server: ' + err);
        showError(aiPanel, aiLabel, aiScore, 'Could not reach the server: ' + err);
      } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze';
      }
    }

    btn.addEventListener('click', runAnalysis);
    textEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.ctrlKey) runAnalysis();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    sentiment_result = analyze(text)
    ai_result = detect(text)

    return jsonify(
        {
            "sentiment": sentiment_result,
            "ai_detection": ai_result,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
