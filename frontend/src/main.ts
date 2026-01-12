interface PromptResponse {
  response: string;
  context: Array<{
    talk_id: string;
    title: string;
    chunk: string;
    score: number;
  }>;
}

interface StatsPayload {
  chunk_size: number;
  overlap_ratio: number;
  top_k: number;
}

const form = document.getElementById("prompt-form") as HTMLFormElement;
const questionInput = document.getElementById("question") as HTMLTextAreaElement;
const answerText = document.getElementById("answer-text") as HTMLPreElement;
const contextList = document.getElementById("context-list") as HTMLUListElement;
const loadingBar = document.getElementById("loading-bar") as HTMLDivElement;
const errorBox = document.getElementById("prompt-error") as HTMLDivElement;
const responseBlock = document.getElementById("prompt-response") as HTMLDivElement;
const statsList = document.getElementById("stats-list") as HTMLDivElement;
const clearAnswerBtn = document.getElementById("clear-answer") as HTMLButtonElement;

function setLoading(isLoading: boolean) {
  if (isLoading) {
    loadingBar.classList.add("active");
    form.querySelectorAll("button, textarea").forEach((node) => (node as HTMLButtonElement | HTMLTextAreaElement).setAttribute("disabled", "true"));
  } else {
    loadingBar.classList.remove("active");
    form.querySelectorAll("button, textarea").forEach((node) => node.removeAttribute("disabled"));
  }
}

function showError(message: string | null) {
  if (!message) {
    errorBox.hidden = true;
    errorBox.textContent = "";
    return;
  }
  errorBox.hidden = false;
  errorBox.textContent = message;
}

function renderContextTree(ctx: PromptResponse["context"][number], autoOpen = false) {
  const item = document.createElement("li");
  item.classList.add("context-item");

  const details = document.createElement("details");
  details.open = autoOpen;

  const summary = document.createElement("summary");
  summary.textContent = `${ctx.title} [${ctx.talk_id}]`;
  details.appendChild(summary);

  const body = document.createElement("div");
  body.classList.add("context-body");

  const dl = document.createElement("dl");

  const entries: Array<[string, string]> = [
    ["Talk ID", ctx.talk_id],
    ["Score", ctx.score.toFixed(4)],
    ["Chunk", ctx.chunk.trim()],
  ];

  entries.forEach(([label, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value;
    dl.append(dt, dd);
  });

  body.appendChild(dl);
  details.appendChild(body);
  item.appendChild(details);
  return item;
}

function renderResponse(data: PromptResponse) {
  responseBlock.style.display = "block";
  answerText.textContent = data.response;
  contextList.innerHTML = "";
  data.context.forEach((ctx, idx) => {
    contextList.appendChild(renderContextTree(ctx, idx === 0));
  });
}

async function submitPrompt(event: SubmitEvent) {
  event.preventDefault();
  const query = questionInput.value.trim();
  if (!query) {
    showError("Please enter a prompt first.");
    return;
  }

  setLoading(true);
  showError(null);

  try {
    const resp = await fetch("/api/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query }),
    });

    if (!resp.ok) {
      throw new Error(`Server responded with ${resp.status}`);
    }

    const data = (await resp.json()) as PromptResponse;
    renderResponse(data);
  } catch (err) {
    console.error(err);
    showError((err as Error).message || "Something went wrong.");
  } finally {
    setLoading(false);
  }
}

function renderStats(stats: StatsPayload) {
  statsList.innerHTML = "";
  const entries: Array<[string, string]> = [
    ["Chunk size", stats.chunk_size.toString()],
    ["Overlap ratio", stats.overlap_ratio.toString()],
    ["Top k", stats.top_k.toString()],
  ];
  entries.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.classList.add("stat-item");
    const labelEl = document.createElement("p");
    labelEl.classList.add("label");
    labelEl.textContent = label;
    const valueEl = document.createElement("p");
    valueEl.classList.add("value");
    valueEl.textContent = value;
    card.append(labelEl, valueEl);
    statsList.appendChild(card);
  });
}

async function fetchStats() {
  try {
    const resp = await fetch("/api/stats");
    if (!resp.ok) {
      throw new Error(`Stats request failed with ${resp.status}`);
    }
    const stats = (await resp.json()) as StatsPayload;
    renderStats(stats);
  } catch (err) {
    console.error(err);
    showError((err as Error).message || "Unable to fetch stats.");
  }
}

function clearAnswer() {
  responseBlock.style.display = "none";
  answerText.textContent = "";
  contextList.innerHTML = "";
  showError(null);
}

form.addEventListener("submit", submitPrompt);
clearAnswerBtn.addEventListener("click", clearAnswer);

fetchStats();
clearAnswer();
setLoading(false);
