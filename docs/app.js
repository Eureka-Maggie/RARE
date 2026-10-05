/* Original case text is rendered locally with Marked and sanitized with DOMPurify. */
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
  let cases = [];
  let activeCase;
  let domain = "2d";
  let view = "excerpt";
  const number = (value) => Number(value).toFixed(2).replace(/\.00$/, "");
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

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
        button.append(
          el("small", "", `CASE ${item.id.toUpperCase()}`),
          el("strong", "", item.label),
          el("span", "case-cn", item.labelZh),
          el(
            "span",
            "case-delta",
            `Overall ${number(item.scores[0].base)} → ${number(item.scores[0].rare)}`,
          ),
        );
        button.querySelector(".case-cn").lang = "zh-CN";
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
      const script = activeCase[variant];
      const target = $(`#${variant}-script`);
      // Preserve generated text, including formatting and production annotations.
      target.innerHTML = DOMPurify.sanitize(
        marked.parse(view === "full" ? script.markdown : script.excerpt),
      );
      target.scrollTop = 0;
      $(`#${variant}-scene-label`).textContent =
        view === "full"
          ? "Complete output · story, cast, scenes & production notes"
          : `Excerpt · ${script.sceneLabel} · full context available in “Full scripts”`;
    });
  }

  function selectCase(id, updateURL = false) {
    const item = cases.find((candidate) => candidate.id === id);
    if (!item) return;
    const domainChanged = domain !== item.domain;
    activeCase = item;
    domain = item.domain;
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
    $("#query").textContent = item.query;
    $("#query-en").textContent = item.queryEn;
    $("#case-id").textContent = item.id.toUpperCase();
    const url = new URL(location.href);
    url.searchParams.set("case", id);
    url.hash = "examples";
    $("#permalink").href = url.href;
    if (updateURL) history.pushState({ case: id }, "", url);
    ["base", "rare"].forEach((variant) => {
      $(`#${variant}-title`).textContent = item[variant].title;
      $(`#${variant}-overall`).replaceChildren(
        document.createTextNode(number(item.scores[0][variant])),
        el("small", "", "/ 100"),
      );
      $(`#${variant}-download`).href = item[variant].download;
      $(`#${variant}-download`).setAttribute(
        "download",
        `${id.replace("/", "-")}-${variant}.md`,
      );
    });
    renderScores();
    renderScripts();
    $("#case-detail").hidden = false;
  }

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
  window.addEventListener("popstate", () =>
    selectCase(new URL(location.href).searchParams.get("case") || cases[0]?.id),
  );

  fetch("assets/cases.json")
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
        "The examples could not be loaded. Please reload the page or ",
      );
      const link = el("a", "", "browse the original scripts on GitHub");
      link.href = "https://github.com/Eureka-Maggie/RARE/tree/main/docs/cases";
      message.append(link, document.createTextNode("."));
      $("#case-list").replaceChildren(message);
    });
})();
