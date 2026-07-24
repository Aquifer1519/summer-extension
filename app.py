"""
Minimal local test server for the sentiment scale.

Run with:
    pip install flask
    python app.py

Then open http://127.0.0.1:5000 in your browser. Type or paste text,
hit Analyze (or press Ctrl+Enter), and see the score update without a
page reload.
"""

from flask import Flask, request, jsonify, render_template_string
from sentiment import analyze

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sentiment Scale Tester</title>
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
    #result {
      margin-top: 24px;
      padding: 16px;
      border-radius: 8px;
      background: #f5f5f5;
      display: none;
    }
    #bar-track {
      position: relative;
      height: 24px;
      background: linear-gradient(to right, #d32f2f, #eee 50%, #2e7d32);
      border-radius: 12px;
      margin: 12px 0;
    }
    #bar-marker {
      position: absolute;
      top: -4px;
      width: 4px;
      height: 32px;
      background: #222;
      border-radius: 2px;
      transform: translateX(-2px);
    }
    #label { font-size: 18px; font-weight: 600; }
    #score { color: #555; font-size: 14px; }
    .error { color: #d32f2f; }
  </style>
</head>
<body>
  <h1>Sentiment Scale Tester</h1>
  <textarea id="text" placeholder="Type or paste text here... (Ctrl+Enter to analyze)"></textarea>
  <br>
  <button id="analyze-btn">Analyze</button>

  <div id="result">
    <div id="label"></div>
    <div id="score"></div>
    <div id="bar-track"><div id="bar-marker"></div></div>
    <div style="font-size:12px;color:#888;display:flex;justify-content:space-between;">
      <span>strongly negative</span><span>neutral</span><span>strongly positive</span>
    </div>
  </div>

  <script>
    const btn = document.getElementById('analyze-btn');
    const textEl = document.getElementById('text');
    const resultEl = document.getElementById('result');
    const labelEl = document.getElementById('label');
    const scoreEl = document.getElementById('score');
    const markerEl = document.getElementById('bar-marker');

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
          labelEl.textContent = 'Error';
          labelEl.className = 'error';
          scoreEl.textContent = data.error;
        } else {
          labelEl.textContent = data.scale_label;
          labelEl.className = '';
          scoreEl.textContent = `score: ${data.scale_score.toFixed(4)}  |  raw: ${JSON.stringify(data.raw_scores)}`;
          // score ranges -1 to 1 -> map to 0%-100% position on the bar
          const pct = ((data.scale_score + 1) / 2) * 100;
          markerEl.style.left = pct + '%';
        }
        resultEl.style.display = 'block';
      } catch (err) {
        labelEl.textContent = 'Error';
        labelEl.className = 'error';
        scoreEl.textContent = 'Could not reach the server: ' + err;
        resultEl.style.display = 'block';
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

    result = analyze(text)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
