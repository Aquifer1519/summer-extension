const API_URL = "http://127.0.0.1:5000/analyze";

const textEl = document.getElementById('text');
const btn = document.getElementById('analyze-btn');
const statusEl = document.getElementById('status');

const sentimentPanel = document.getElementById('sentiment-panel');
const sentimentLabel = document.getElementById('sentiment-label');
const sentimentScore = document.getElementById('sentiment-score');
const sentimentMarker = document.getElementById('sentiment-marker');

const aiPanel = document.getElementById('ai-panel');
const aiLabel = document.getElementById('ai-label');
const aiScore = document.getElementById('ai-score');
const aiMarker = document.getElementById('ai-marker');

function aiLabelForScore(score) {
  if (score >= 0.7) return 'likely AI';
  if (score >= 0.4) return 'uncertain';
  return 'likely human';
}

function showError(panel, labelEl, scoreEl, message) {
  labelEl.textContent = 'Error';
  labelEl.className = 'label error';
  scoreEl.textContent = message;
  panel.style.display = 'block';
}

async function runAnalysis() {
  const text = textEl.value.trim();
  if (!text) return;

  btn.disabled = true;
  btn.textContent = 'Analyzing...';
  statusEl.textContent = '';

  try {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      throw new Error(`Server responded with ${res.status}`);
    }

    const data = await res.json();

    if (data.error) {
      showError(sentimentPanel, sentimentLabel, sentimentScore, data.error);
      showError(aiPanel, aiLabel, aiScore, data.error);
    } else {
      sentimentLabel.textContent = data.sentiment.scale_label;
      sentimentLabel.className = 'label';
      sentimentScore.textContent = `score: ${data.sentiment.scale_score.toFixed(4)}`;
      const sentPct = ((data.sentiment.scale_score + 1) / 2) * 100;
      sentimentMarker.style.left = sentPct + '%';
      sentimentPanel.style.display = 'block';

      const aiVal = data.ai_detection.ai_likelihood;
      aiLabel.textContent = aiLabelForScore(aiVal);
      aiLabel.className = 'label';
      aiScore.textContent = `AI likelihood: ${aiVal.toFixed(4)}`;
      const aiPct = aiVal * 100;
      aiMarker.style.left = aiPct + '%';
      aiPanel.style.display = 'block';
    }
  } catch (err) {
    statusEl.textContent = 'Could not reach the local server. Is app.py running on port 5000?';
    statusEl.className = 'status error';
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze';
  }
}

btn.addEventListener('click', runAnalysis);

// On popup open, try to prefill with whatever text is currently selected
// on the active page, so the common flow (select text -> click extension
// icon) doesn't require manual copy-paste.
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    });
    if (result && result.trim()) {
      textEl.value = result.trim();
    }
  } catch (err) {
    // Fails silently on chrome:// pages, the extensions page, etc. --
    // not an error worth surfacing, just leave the textarea empty.
    console.log('Could not read page selection:', err);
  }
});
