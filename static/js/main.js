/* LLaDA-UI project page — shared page behaviour: the denoising headline,
 * scroll reveal, count-up numbers, BibTeX copy, footer year. */

(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function growCharts(root) {
    if (window.LladaPage && window.LladaPage.growCharts) window.LladaPage.growCharts(root);
  }

  /* ======================= Denoising headline =======================
   * Elements with [data-denoise] start fully masked and resolve the way the
   * model decodes: the text is split into contiguous blocks (data-blocks,
   * default 1) that resolve strictly left to right, while tokens inside a
   * block commit in parallel, a few per step, in no fixed order. */

  function wrapTokens(el, mode) {
    var tokens = [];
    function walk(node) {
      if (node.nodeType === Node.TEXT_NODE) {
        var parts = mode === "chars"
          ? node.textContent.split("")
          : node.textContent.split(/(\s+)/);
        var frag = document.createDocumentFragment();
        parts.forEach(function (part) {
          if (!part) return;
          if (/^\s+$/.test(part)) {
            frag.appendChild(document.createTextNode(part));
          } else {
            var span = document.createElement("span");
            span.className = "dn masked";
            span.textContent = part;
            frag.appendChild(span);
            tokens.push(span);
          }
        });
        node.parentNode.replaceChild(frag, node);
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        Array.prototype.slice.call(node.childNodes).forEach(walk);
      }
    }
    Array.prototype.slice.call(el.childNodes).forEach(walk);
    return tokens;
  }

  function denoise(el) {
    var tokens = wrapTokens(el, el.dataset.denoise || "words");
    if (!tokens.length) return;
    var nBlocks = Math.max(1, parseInt(el.dataset.blocks, 10) || 1);
    var startDelay = parseInt(el.dataset.delay, 10) || 250;
    var blockDur = parseInt(el.dataset.blockDur, 10) || 900;

    var perBlock = Math.ceil(tokens.length / nBlocks);
    for (var b = 0; b < nBlocks; b++) {
      var block = tokens.slice(b * perBlock, (b + 1) * perBlock);
      /* shuffle: commits inside a block have no fixed order */
      for (var i = block.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var tmp = block[i]; block[i] = block[j]; block[j] = tmp;
      }
      block.forEach(function (span, k) {
        /* ease-in: few commits at first, then accelerating — like
           confidence-based parallel unmasking */
        var p = block.length > 1 ? k / (block.length - 1) : 1;
        var delay = startDelay + b * blockDur + Math.pow(p, 0.7) * blockDur;
        setTimeout(function () { span.classList.remove("masked"); }, delay);
      });
    }
  }

  if (!reduceMotion) {
    document.querySelectorAll("[data-denoise]").forEach(denoise);
  }

  /* ======================= Count-up numbers ======================= */

  function countUp(node) {
    var target = parseFloat(node.dataset.target);
    if (isNaN(target) || reduceMotion) { node.textContent = node.dataset.target; return; }
    var start = null;
    var dur = 1100;
    function tick(ts) {
      if (!start) start = ts;
      var p = Math.min(1, (ts - start) / dur);
      var eased = 1 - Math.pow(1 - p, 3);
      node.textContent = (target * eased).toFixed(2);
      if (p < 1) requestAnimationFrame(tick);
      else node.textContent = node.dataset.target;
    }
    requestAnimationFrame(tick);
  }

  /* ======================= Scroll reveal ======================= */

  var revealTargets = document.querySelectorAll("[data-reveal]");
  var countTargets = document.querySelectorAll(".count-up");

  function activate(elmt) {
    elmt.classList.add("revealed");
    growCharts(elmt);
    elmt.querySelectorAll(".count-up").forEach(function (n) {
      if (!n.dataset.done) { n.dataset.done = "1"; countUp(n); }
    });
  }

  if (reduceMotion || !("IntersectionObserver" in window)) {
    revealTargets.forEach(activate);
    growCharts(document);
    countTargets.forEach(function (n) { n.textContent = n.dataset.target; });
  } else {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            activate(entry.target);
            observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }
    );
    revealTargets.forEach(function (elmt) { observer.observe(elmt); });

    var countObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          var n = entry.target;
          if (!n.dataset.done) { n.dataset.done = "1"; countUp(n); }
          countObserver.unobserve(n);
        }
      });
    }, { threshold: 0.4 });
    countTargets.forEach(function (n) { countObserver.observe(n); });
  }

  /* ======================= BibTeX copy ======================= */

  var copyBtn = document.getElementById("copy-bibtex");
  var bibtex = document.getElementById("bibtex-text");
  if (copyBtn && bibtex) {
    copyBtn.addEventListener("click", function () {
      var done = function () {
        copyBtn.textContent = "Copied ✓";
        setTimeout(function () { copyBtn.textContent = "Copy"; }, 1800);
      };
      var fallback = function () {
        var range = document.createRange();
        range.selectNodeContents(bibtex);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand("copy");
        sel.removeAllRanges();
        done();
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(bibtex.textContent).then(done).catch(fallback);
      } else {
        fallback();
      }
    });
  }

  var year = document.getElementById("year");
  if (year) { year.textContent = String(new Date().getFullYear()); }
})();
