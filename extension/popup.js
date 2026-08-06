const API_URL = "http://127.0.0.1:5000/analyze";
const STORAGE_KEY = "vibeCheckState";

const textEl = document.getElementById("text");
const btn = document.getElementById("analyze-btn");
const statusEl = document.getElementById("status");

const sentimentPanel = document.getElementById("sentiment-panel");
const sentimentLabel = document.getElementById("sentiment-label");
const sentimentScore = document.getElementById("sentiment-score");
const sentimentMarker = document.getElementById("sentiment-marker");

const aiPanel = document.getElementById("ai-panel");
const aiLabel = document.getElementById("ai-label");
const aiScore = document.getElementById("ai-score");
const aiMarker = document.getElementById("ai-marker");

let isResetMode = false;

function setButtonMode(resetMode) {
  isResetMode = resetMode;
  btn.dataset.mode = resetMode ? "reset" : "scan";
  btn.textContent = resetMode ? "New Scan" : "Scan";
}

function clearAnalysis() {
  localStorage.removeItem(STORAGE_KEY);
  textEl.value = "";
  sentimentPanel.style.display = "none";
  aiPanel.style.display = "none";
  statusEl.textContent = "";
  statusEl.className = "status";
  setButtonMode(false);
}

function aiLabelForScore(score) {
  if (score >= 0.7) return "likely AI";
  if (score >= 0.4) return "uncertain";
  return "likely human";
}

function getSavedState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (err) {
    console.warn("Could not parse saved state", err);
    return null;
  }
}

function saveState(state = null) {
  if (!state) {
    state = {
      text: textEl.value,
      sentiment: {
        label: sentimentLabel.textContent,
        score: sentimentScore.textContent,
        markerLeft: sentimentMarker.style.left,
        visible: sentimentPanel.style.display === "block",
        labelClass: sentimentLabel.className,
      },
      ai: {
        label: aiLabel.textContent,
        score: aiScore.textContent,
        markerLeft: aiMarker.style.left,
        visible: aiPanel.style.display === "block",
        labelClass: aiLabel.className,
      },
      status: {
        text: statusEl.textContent,
        className: statusEl.className,
      },
    };
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function restoreState() {
  const state = getSavedState();
  if (!state) return false;

  textEl.value = state.text || "";

  if (state.sentiment?.visible) {
    sentimentLabel.textContent = state.sentiment.label || "";
    sentimentLabel.className = state.sentiment.labelClass || "label";
    sentimentScore.textContent = state.sentiment.score || "";
    sentimentMarker.style.left = state.sentiment.markerLeft || "0%";
    sentimentPanel.style.display = "block";
  } else {
    sentimentPanel.style.display = "none";
  }

  if (state.ai?.visible) {
    aiLabel.textContent = state.ai.label || "";
    aiLabel.className = state.ai.labelClass || "label";
    aiScore.textContent = state.ai.score || "";
    aiMarker.style.left = state.ai.markerLeft || "0%";
    aiPanel.style.display = "block";
  } else {
    aiPanel.style.display = "none";
  }

  statusEl.textContent = state.status?.text || "";
  statusEl.className = state.status?.className || "status";
  if (!statusEl.textContent) {
    statusEl.className = "status";
  }

  const shouldReset =
    state.sentiment?.visible || state.ai?.visible || !!state.status?.text;
  setButtonMode(shouldReset);

  return true;
}

function showError(panel, labelEl, scoreEl, message) {
  labelEl.textContent = "Error";
  labelEl.className = "label error";
  scoreEl.textContent = message;
  panel.style.display = "block";
}

async function runAnalysis() {
  if (isResetMode) {
    clearAnalysis();
    return;
  }

  const text = textEl.value.trim();
  if (!text) return;

  btn.disabled = true;
  btn.textContent = "Analyzing...";
  statusEl.textContent = "";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) {
      throw new Error(`Server responded with ${res.status}`);
    }

    const data = await res.json();

    if (data.error) {
      showError(sentimentPanel, sentimentLabel, sentimentScore, data.error);
      showError(aiPanel, aiLabel, aiScore, data.error);
      setButtonMode(true);
      saveState();
    } else {
      sentimentLabel.textContent = data.sentiment.scale_label;
      sentimentLabel.className = "label";
      sentimentScore.textContent = `score: ${data.sentiment.scale_score.toFixed(4)}`;
      const sentPct = ((data.sentiment.scale_score + 1) / 2) * 100;
      sentimentMarker.style.left = sentPct + "%";
      sentimentPanel.style.display = "block";

      const oppVal = data.ai_detection.ai_likelihood;
      const aiVal = 1 - oppVal;

      aiLabel.textContent = aiLabelForScore(oppVal);
      aiLabel.className = "label";
      aiScore.textContent = `AI likelihood: ${oppVal.toFixed(4)}`;
      const aiPct = aiVal * 100;
      aiMarker.style.left = aiPct + "%";
      aiPanel.style.display = "block";

      setButtonMode(true);
      saveState();
    }
  } catch (err) {
    statusEl.textContent =
      "Could not reach the local server. Is app.py running on port 5000?";
    statusEl.className = "status error";
    console.error(err);
    setButtonMode(true);
    saveState();
  } finally {
    btn.disabled = false;
    if (!isResetMode) {
      setButtonMode(false);
    }
  }
}

btn.addEventListener("click", runAnalysis);
textEl.addEventListener("input", saveState);

// On popup open, restore the last analysis state if available.
// If no prior state exists, try to prefill with whatever text is
// currently selected on the active page.
document.addEventListener("DOMContentLoaded", async () => {
  const restored = restoreState();
  if (restored) {
    return;
  }
  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });
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
    console.log("Could not read page selection:", err);
  }
});
