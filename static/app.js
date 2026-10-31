// Twitter Post Emotion Analysis Application Logic

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const sentenceInput = document.getElementById("sentenceInput");
  const charCounter = document.getElementById("charCounter");
  const detectBtn = document.getElementById("detectBtn");
  const toggleTraceBtn = document.getElementById("toggleTraceBtn");
  const traceDrawer = document.getElementById("traceDrawer");

  // Trace Elements
  const traceRaw = document.getElementById("traceRaw");
  const traceEmojis = document.getElementById("traceEmojis");
  const traceExpanded = document.getElementById("traceExpanded");
  const traceNegated = document.getElementById("traceNegated");
  const traceCleaned = document.getElementById("traceCleaned");

  // Hero Card Elements
  const heroCard = document.getElementById("heroCard");
  const heroEmoji = document.getElementById("heroEmoji");
  const heroEmotionTitle = document.getElementById("heroEmotionTitle");
  const heroConfidencePill = document.getElementById("heroConfidencePill");
  const heroConfidenceSub = document.getElementById("heroConfidenceSub");
  const backendBadge = document.getElementById("backendBadge");

  // Breakdown List
  const distributionList = document.getElementById("distributionList");

  // History Elements
  const historyTableBody = document.getElementById("historyTableBody");
  const historyCount = document.getElementById("historyCount");
  const historySearch = document.getElementById("historySearch");
  const historyFilter = document.getElementById("historyFilter");
  const exportHistoryBtn = document.getElementById("exportHistoryBtn");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  // Emoji and Label Dictionary
  const EMOJI_MAP = {
    joy: "😊",
    sadness: "😢",
    anger: "😡",
    fear: "😱",
    surprise: "😲",
    disgust: "🤢",
    neutral: "😐"
  };

  const EMOTION_TITLES = {
    joy: "JOY & HAPPINESS",
    sadness: "SADNESS & GRIEF",
    anger: "ANGER & FRUSTRATION",
    fear: "FEAR & ANXIETY",
    surprise: "SURPRISE & AMAZEMENT",
    disgust: "DISGUST & REVULSION",
    neutral: "NEUTRAL / FACTUAL"
  };

  // Local Storage Key
  const HISTORY_STORAGE_KEY = "ekman_tweet_emotion_history";

  // 1. Character Counter
  sentenceInput.addEventListener("input", () => {
    const len = sentenceInput.value.length;
    charCounter.textContent = `${len} / 280 chars`;
  });

  // 2. Preset Chips Click
  document.querySelectorAll(".preset-chips .chip").forEach(chip => {
    chip.addEventListener("click", () => {
      sentenceInput.value = chip.getAttribute("data-text");
      sentenceInput.dispatchEvent(new Event("input"));
      predictEmotion();
    });
  });

  // 3. Toggle Preprocessing Trace Drawer
  toggleTraceBtn.addEventListener("click", () => {
    traceDrawer.classList.toggle("hidden");
  });

  // 4. Keyboard Shortcut (Ctrl+Enter or Cmd+Enter)
  sentenceInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      predictEmotion();
    }
  });

  // 5. Detect Emotion Execution
  async function predictEmotion() {
    const text = sentenceInput.value.strip ? sentenceInput.value.strip() : sentenceInput.value.trim();
    if (!text) {
      alert("Please paste or type a tweet first!");
      return;
    }

    // Set Loading State
    detectBtn.disabled = true;
    detectBtn.querySelector(".btn-text").textContent = "Analyzing Tweet...";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      renderResults(data);
      saveToHistory(data);
    } catch (err) {
      console.error("Prediction Error:", err);
      alert(`Error running prediction: ${err.message}`);
    } finally {
      detectBtn.disabled = false;
      detectBtn.querySelector(".btn-text").textContent = "Analyze Tweet";
    }
  }

  detectBtn.addEventListener("click", predictEmotion);

  // 6. Render Results to UI
  function renderResults(data) {
    const primary = data.primary_emotion.toLowerCase();
    const confidencePct = data.confidence_percentage;
    const emoji = EMOJI_MAP[primary] || "😐";

    // Hero Card Styling
    heroCard.className = `hero-card emotion-${primary}`;
    heroEmoji.textContent = emoji;
    heroEmotionTitle.textContent = primary.toUpperCase();
    heroConfidencePill.textContent = `${confidencePct}% Confidence`;
    heroConfidenceSub.textContent = EMOTION_TITLES[primary] || "Detected Emotion";

    if (data.backend) {
      backendBadge.textContent = `Engine: ${data.backend}`;
    }

    // Preprocessing Trace
    const trace = data.preprocessing_details || {};
    traceRaw.textContent = trace.raw_text || data.text;
    traceEmojis.textContent = (trace.emojis_found && trace.emojis_found.length > 0) ? trace.emojis_found.join(" ") : "None";
    traceExpanded.textContent = trace.expanded_contractions || "-";
    traceNegated.textContent = trace.negations_tagged || "-";
    traceCleaned.textContent = trace.final_cleaned || data.cleaned_text;

    // Taxonomy Breakdown Bars
    distributionList.innerHTML = "";
    data.breakdown.forEach(item => {
      const emo = item.emotion.toLowerCase();
      const pct = item.percentage;
      const emoEmoji = EMOJI_MAP[emo] || "•";

      const itemEl = document.createElement("div");
      itemEl.className = "dist-item";
      itemEl.innerHTML = `
        <div class="dist-header">
          <span class="dist-name">${emoEmoji} ${emo.toUpperCase()}</span>
          <span class="dist-pct">${pct}%</span>
        </div>
        <div class="dist-bar-track">
          <div class="dist-bar-fill fill-${emo}" style="width: ${pct}%"></div>
        </div>
      `;
      distributionList.appendChild(itemEl);
    });
  }

  // 7. Local History Management
  function getHistory() {
    try {
      const stored = localStorage.getItem(HISTORY_STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  function saveToHistory(data) {
    const history = getHistory();
    const newEntry = {
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      text: data.text,
      primary: data.primary_emotion,
      confidence: data.confidence_percentage,
      breakdown: data.breakdown
    };

    history.unshift(newEntry);
    // Keep max 50 entries
    if (history.length > 50) history.pop();

    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    renderHistoryTable();
  }

  function renderHistoryTable() {
    const history = getHistory();
    const searchTerm = historySearch.value.toLowerCase().trim();
    const filterEmotion = historyFilter.value.toLowerCase();

    const filtered = history.filter(item => {
      const matchesSearch = item.text.toLowerCase().includes(searchTerm);
      const matchesFilter = filterEmotion === "all" || item.primary.toLowerCase() === filterEmotion;
      return matchesSearch && matchesFilter;
    });

    historyCount.textContent = `${history.length} tweets analyzed locally`;
    historyTableBody.innerHTML = "";

    if (filtered.length === 0) {
      historyTableBody.innerHTML = `
        <tr>
          <td colspan="5" class="empty-row">No tweet analysis matching criteria.</td>
        </tr>
      `;
      return;
    }

    filtered.forEach(item => {
      const emo = item.primary.toLowerCase();
      const emoji = EMOJI_MAP[emo] || "•";
      const tr = document.createElement("tr");

      tr.innerHTML = `
        <td>${item.timestamp}</td>
        <td><strong>"${escapeHtml(item.text)}"</strong></td>
        <td><span class="chip fill-${emo}" style="display:inline-block;">${emoji} ${emo.toUpperCase()}</span></td>
        <td><strong>${item.confidence}%</strong></td>
        <td>
          <button class="btn btn-sm btn-outline reanalyze-btn" data-text="${escapeHtml(item.text)}">Re-analyze</button>
        </td>
      `;

      historyTableBody.appendChild(tr);
    });

    // Re-analyze click
    document.querySelectorAll(".reanalyze-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        sentenceInput.value = btn.getAttribute("data-text");
        sentenceInput.dispatchEvent(new Event("input"));
        predictEmotion();
      });
    });
  }

  function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Search and Filter Listeners
  historySearch.addEventListener("input", renderHistoryTable);
  historyFilter.addEventListener("change", renderHistoryTable);

  // Clear History
  clearHistoryBtn.addEventListener("click", () => {
    if (confirm("Are you sure you want to clear all tweet analysis history?")) {
      localStorage.removeItem(HISTORY_STORAGE_KEY);
      renderHistoryTable();
    }
  });

  // Export History JSON
  exportHistoryBtn.addEventListener("click", () => {
    const history = getHistory();
    if (history.length === 0) {
      alert("History is empty!");
      return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `tweet_emotion_history_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  });

  // Initial Load History Table
  renderHistoryTable();
});
