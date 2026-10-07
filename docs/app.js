/* Source outputs and English translations are rendered with Marked and DOMPurify. */
"use strict";
(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const domains = ["2d", "3d", "live"];
  const metrics = [
    "Query fulfillment",
    "Narrative coherence",
    "Cinematic executability",
    "Dramatic effectiveness",
  ];
  const preferenceKey = "rare-script-language";
  let cases = [];
  let activeCase;
  let domain = "2d";
  let view = "excerpt";
  let savedLanguage;
  try {
    savedLanguage = localStorage.getItem(preferenceKey);
  } catch {
    /* Storage is optional. */
  }
  const validLanguage = (value) => ["en", "zh"].includes(value);
  const requestedLanguage = new URL(location.href).searchParams.get("lang");
  let language = validLanguage(requestedLanguage)
    ? requestedLanguage
    : validLanguage(savedLanguage)
      ? savedLanguage
      : "en";
  const number = (value) => Number(value).toFixed(2).replace(/\.00$/, "");
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const currentScript = (variant) =>
    language === "en" ? activeCase[variant].en : activeCase[variant];

  function caseURL() {
    const url = new URL(location.href);
    url.searchParams.set("case", activeCase.id);
    url.searchParams.set("lang", language);
    url.hash = "examples";
    return url;
  }

  function updateLanguageControls() {
    $$(".language-switch button").forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.language === language),
      ),
    );
    $("#language-note").textContent =
      language === "en"
        ? "English translations of the Chinese outputs. Scores evaluate the original scripts."
        : "中文原始输出。评分基于原文；切换 EN 可阅读对应的英文译文。";
    $("#language-note").lang = language === "en" ? "en" : "zh-CN";
  }

  function renderChoices() {
    const list = $("#case-list");
    list.replaceChildren();
    cases
      .filter((item) => item.domain === domain)
      .forEach((item) => {
        const button = el("button", "case-choice");
        button.type = "button";
        button.dataset.case = item.id;
        button.setAttribute("aria-pressed", String(item.id === activeCase.id));
        const title = el(
          "strong",
          "",
          language === "en" ? item.label : item.labelZh,
        );
        title.lang = language === "en" ? "en" : "zh-CN";
        button.append(
          el("small", "", item.id.toUpperCase()),
          title,
          el(
            "span",
            "case-delta",
            `${number(item.scores[0].base)} → ${number(item.scores[0].rare)}`,
          ),
        );
        button.addEventListener("click", () => selectCase(item.id, true));
        list.append(button);
      });
  }

  function renderScores() {
    const scores = $("#scores");
    scores.replaceChildren();
    activeCase.scores.slice(1).forEach((metric, index) => {
      const block = el("div", "metric");
      block.append(el("p", "metric-title", metrics[index]));
      ["base", "rare"].forEach((variant) => {
        const row = el("div", `metric-row ${variant}`);
        row.setAttribute("role", "img");
        row.setAttribute(
          "aria-label",
          `${variant === "base" ? "Base" : "RARE"}: ${number(metric[variant])} out of 100`,
        );
        const track = el("span", "bar-track");
        const bar = el("span", "bar");
        bar.style.width = `${metric[variant]}%`;
        track.append(bar);
        row.append(track, el("span", "value", number(metric[variant])));
        block.append(row);
      });
      scores.append(block);
    });
  }

  function renderScripts() {
    $$(".view-switch button").forEach((button) =>
      button.setAttribute("aria-pressed", String(button.dataset.view === view)),
    );
    ["base", "rare"].forEach((variant) => {
      const script = currentScript(variant);
      const target = $(`#${variant}-script`);
      target.innerHTML = DOMPurify.sanitize(
        marked.parse(view === "full" ? script.markdown : script.excerpt),
      );
      target.lang = language === "en" ? "en" : "zh-CN";
      target.scrollTop = 0;
      $(`#${variant}-title`).textContent = script.title;
      $(`#${variant}-title`).lang = target.lang;
      $(`#${variant}-scene-label`).textContent =
        view === "full"
          ? "Complete script, including cast and production notes"
          : `Selected scenes: ${script.sceneLabel}`;
      $(`#${variant}-download`).href = script.download;
      $(`#${variant}-download`).setAttribute(
        "download",
        `${activeCase.id.replace("/", "-")}-${variant}-${language}.md`,
      );
    });
  }

  function renderPrompt() {
    $("#query").textContent =
      language === "en" ? activeCase.queryEn : activeCase.query;
    $("#query").lang = language === "en" ? "en" : "zh-CN";
    $("#permalink").href = caseURL().href;
  }

  function selectCase(id, updateURL = false) {
    const item = cases.find((candidate) => candidate.id === id);
    if (!item) return;
    const domainChanged = domain !== item.domain;
    activeCase = item;
    domain = item.domain;
    $("#examples").dataset.activeDomain = domain;
    $$(".domain-tabs button").forEach((button) => {
      const selected = button.dataset.domain === domain;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
    $("#case-browser").setAttribute("aria-labelledby", `tab-${domain}`);
    if (domainChanged || !$("#case-list button")) renderChoices();
    else
      $$(".case-choice").forEach((button) =>
        button.setAttribute("aria-pressed", String(button.dataset.case === id)),
      );
    $("#case-id").textContent = item.id.toUpperCase();
    renderPrompt();
    if (updateURL) history.pushState({ case: id, language }, "", caseURL());
    ["base", "rare"].forEach((variant) => {
      $(`#${variant}-overall`).replaceChildren(
        document.createTextNode(number(item.scores[0][variant])),
        el("small", "", "/ 100"),
      );
    });
    renderScores();
    renderScripts();
    $("#case-detail").hidden = false;
  }

  function setLanguage(next, updateURL = true) {
    if (!validLanguage(next)) return;
    language = next;
    try {
      localStorage.setItem(preferenceKey, next);
    } catch {
      /* Storage is optional. */
    }
    updateLanguageControls();
    if (!activeCase) return;
    renderChoices();
    renderPrompt();
    renderScripts();
    if (updateURL)
      history.pushState({ case: activeCase.id, language }, "", caseURL());
  }

  $$(".language-switch button").forEach((button) =>
    button.addEventListener("click", () =>
      setLanguage(button.dataset.language),
    ),
  );
  $$(".domain-tabs button").forEach((button, index) => {
    button.addEventListener("click", () => {
      const item = cases.find(
        (candidate) => candidate.domain === button.dataset.domain,
      );
      if (item) selectCase(item.id, true);
    });
    button.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % domains.length;
      else if (event.key === "ArrowLeft")
        next = (index + domains.length - 1) % domains.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = domains.length - 1;
      else return;
      event.preventDefault();
      const target = $$(".domain-tabs button")[next];
      target.focus();
      target.click();
    });
  });
  $$(".view-switch button").forEach((button) =>
    button.addEventListener("click", () => {
      if (!activeCase) return;
      view = button.dataset.view;
      renderScripts();
    }),
  );
  window.addEventListener("popstate", () => {
    const url = new URL(location.href);
    setLanguage(url.searchParams.get("lang") || "en", false);
    selectCase(url.searchParams.get("case") || cases[0]?.id);
  });
  updateLanguageControls();
  fetch("assets/cases.json?v=2")
    .then((response) => {
      if (!response.ok) throw new Error(`Case data: HTTP ${response.status}`);
      return response.json();
    })
    .then((data) => {
      cases = data.cases;
      const requested = new URL(location.href).searchParams.get("case");
      selectCase(
        cases.some((item) => item.id === requested) ? requested : cases[0].id,
      );
    })
    .catch((error) => {
      console.error(error);
      const message = el(
        "p",
        "",
        "The examples could not be loaded. Reload the page or ",
      );
      const link = el("a", "", "read the scripts on GitHub");
      link.href = "https://github.com/Eureka-Maggie/RARE/tree/main/docs/cases";
      message.append(link, document.createTextNode("."));
      $("#case-list").replaceChildren(message);
    });
})();
