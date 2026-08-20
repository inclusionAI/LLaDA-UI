/* LLaDA-UI project page — results charts: renderer, domain tabs, bar
 * tooltip, and lazy loading of the full comparison tables.
 * Chart data lives in results-data.js (window.LLADA_CHARTS). */

(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var CHARTS = window.LLADA_CHARTS || {};

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function fmt(value, spec) {
    var digits = spec.digits === undefined ? 1 : spec.digits;
    return value.toFixed(digits) + (spec.suffix || "");
  }

  function renderChart(spec) {
    var card = el("div", "chart-card");

    var top = el("div", "chart-top");
    top.appendChild(el("span", "chart-cat", spec.cat));
    top.appendChild(el("span", "chart-metric", spec.metric));
    card.appendChild(top);
    card.appendChild(el("h4", "chart-title", spec.title));

    var plot = el("div", "plot");

    [25, 50, 75, 100].forEach(function (frac) {
      var line = el("div", "gridline");
      line.style.bottom = frac + "%";
      var lbl = el("i", null, String(Math.round(spec.max * frac / 100)));
      line.appendChild(lbl);
      plot.appendChild(line);
    });

    var bars = el("div", "bars");
    var names = el("div", "chart-names");

    spec.bars.forEach(function (b) {
      var label = (b.bound ? "<" : "") + fmt(b.value, spec);
      var slot = el("div", "bar-slot" + (b.kind === "ours" ? " is-ours" : ""));
      slot.tabIndex = 0;
      slot.setAttribute("role", "img");
      slot.setAttribute("aria-label", b.model + " (" + b.org + "): " + label + " on " + spec.title);
      slot.dataset.model = b.model;
      slot.dataset.org = b.org;
      slot.dataset.value = label;
      if (b.note) slot.dataset.note = b.note;

      var bar = el("div", "bar " + b.kind + (b.bound ? " bound" : ""));
      bar.style.setProperty("--h", (b.value / spec.max * 100) + "%");

      var val = el("span", "bar-val", label);
      bar.appendChild(val);
      slot.appendChild(bar);
      bars.appendChild(slot);

      var nameSlot = el("div", "name-slot" + (b.kind === "ours" ? " is-ours" : ""));
      nameSlot.appendChild(el("div", "bar-name", b.name || b.model));
      nameSlot.appendChild(el("div", "bar-sub", b.sub || " "));
      names.appendChild(nameSlot);
    });

    plot.appendChild(bars);
    card.appendChild(plot);
    card.appendChild(names);
    if (spec.foot) card.appendChild(el("p", "chart-foot", spec.foot));
    return card;
  }

  Object.keys(CHARTS).forEach(function (key) {
    var host = document.querySelector('[data-charts="' + key + '"]');
    if (!host) return;
    CHARTS[key].forEach(function (spec) { host.appendChild(renderChart(spec)); });
  });

  function growCharts(root) {
    (root || document).querySelectorAll(".chart-card:not(.grown)").forEach(function (card) {
      if (card.closest("[hidden]")) return;
      if (reduceMotion) { card.classList.add("grown"); return; }
      requestAnimationFrame(function () {
        requestAnimationFrame(function () { card.classList.add("grown"); });
      });
    });
  }

  window.LladaPage = window.LladaPage || {};
  window.LladaPage.growCharts = growCharts;

  /* ======================= Domain tabs ======================= */

  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab-btn"));
  tabs.forEach(function (tab) {
    tab.addEventListener("click", function () {
      tabs.forEach(function (t) {
        var selected = t === tab;
        t.setAttribute("aria-selected", selected ? "true" : "false");
        var panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !selected;
      });
      var active = document.getElementById(tab.getAttribute("aria-controls"));
      if (active) growCharts(active);
    });
  });

  /* ======================= Bar tooltip ======================= */

  var tooltip = document.getElementById("chart-tooltip");

  function showTooltip(slot) {
    if (!tooltip) return;
    var html = '<div class="tt-model">' + slot.dataset.model + "</div>" +
      '<div>' + slot.dataset.org + ' · <span class="tt-val">' + slot.dataset.value + "</span></div>";
    if (slot.dataset.note) html += '<div class="tt-note">' + slot.dataset.note + "</div>";
    tooltip.innerHTML = html;
    var rect = slot.getBoundingClientRect();
    tooltip.classList.add("show");
    var ttRect = tooltip.getBoundingClientRect();
    var left = rect.left + rect.width / 2 - ttRect.width / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - ttRect.width - 8));
    var top = rect.top - ttRect.height - 10;
    if (top < 8) top = rect.bottom + 10;
    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }

  function hideTooltip() {
    if (tooltip) tooltip.classList.remove("show");
  }

  document.addEventListener("mouseover", function (e) {
    var slot = e.target.closest && e.target.closest(".bar-slot");
    if (slot) showTooltip(slot);
  });
  document.addEventListener("mouseout", function (e) {
    if (e.target.closest && e.target.closest(".bar-slot")) hideTooltip();
  });
  document.addEventListener("focusin", function (e) {
    var slot = e.target.closest && e.target.closest(".bar-slot");
    if (slot) showTooltip(slot);
  });
  document.addEventListener("focusout", hideTooltip);
  window.addEventListener("scroll", hideTooltip, { passive: true });

  /* ======================= Lazy table partials ======================= */

  document.querySelectorAll("details.full-table[data-src]").forEach(function (details) {
    details.addEventListener("toggle", function () {
      if (!details.open || details.dataset.state) return;
      details.dataset.state = "loading";
      fetch(details.dataset.src)
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.text();
        })
        .then(function (html) {
          details.insertAdjacentHTML("beforeend", html);
          details.dataset.state = "done";
        })
        .catch(function () {
          details.dataset.state = "";
          var old = details.querySelector(".table-load-error");
          if (old) old.remove();
          details.insertAdjacentHTML("beforeend",
            '<p class="table-load-error">Could not load the table — close and reopen to retry.</p>');
        });
    });
  });
})();
