/* LLaDA-UI project page — benchmark chart data.
 *
 * ── HOW RESULTS ARE ENCODED ───────────────────────────────────────────────
 * All chart numbers live in the CHARTS object below. Each bar is
 *   { model, name, sub, org, value, kind, bound?, note? }
 *   - model: full canonical name, shown in the hover tooltip
 *   - name : first label line under the bar (short, never wraps)
 *   - sub  : second label line (variant / size), may be ""
 *   - value: number
 *   - kind : "ours" | "diff" (other diffusion models) | "base" (AR models)
 *   - bound: true → the report only gives an upper bound; rendered "<value"
 *            with a hatched bar
 *   - note : optional footnote shown in the hover tooltip
 * Per-chart options: max (y-axis top), digits (label decimals, default 1),
 * suffix (appended to the label, e.g. " s"), foot (footnote line).
 * Numbers mirror the final LLaDA-UI technical report exactly.
 * ──────────────────────────────────────────────────────────────────────────
 */
(function () {
  "use strict";

  var OURS = function (v, extra) {
    var b = { model: "LLaDA-UI", name: "LLaDA-UI", sub: "16.7B MoE", org: "ours · diffusion MoE", value: v, kind: "ours" };
    if (extra) Object.keys(extra).forEach(function (k) { b[k] = extra[k]; });
    return b;
  };
  var Q25 = function (v, extra) {
    var b = { model: "Qwen2.5-VL-7B", name: "Qwen2.5-VL", sub: "7B", org: "autoregressive VLM", value: v, kind: "base" };
    if (extra) Object.keys(extra).forEach(function (k) { b[k] = extra[k]; });
    return b;
  };
  var Q3 = function (v, extra) {
    var b = { model: "Qwen3-VL-8B", name: "Qwen3-VL", sub: "8B", org: "autoregressive VLM", value: v, kind: "base" };
    if (extra) Object.keys(extra).forEach(function (k) { b[k] = extra[k]; });
    return b;
  };
  var Q35 = function (v) {
    return { model: "Qwen3.5-9B", name: "Qwen3.5", sub: "9B", org: "autoregressive VLM", value: v, kind: "base" };
  };
  var LLADAV = function (v) {
    return { model: "LLaDA-V-8B", name: "LLaDA-V", sub: "8B", org: "diffusion VLM", value: v, kind: "diff" };
  };
  var SDAR = function (v) {
    return { model: "SDAR-VL-8B", name: "SDAR-VL", sub: "8B", org: "diffusion VLM", value: v, kind: "diff" };
  };

  var CHARTS = {
    agent: [
      {
        cat: "Grounding", title: "ScreenSpot-V2", metric: "Point-in-box accuracy (%)", max: 100,
        bars: [OURS(90.7), Q25(86.0), Q3(93.0), Q35(94.0)]
      },
      {
        cat: "Grounding", title: "ScreenSpot-Pro", metric: "Point-in-box accuracy (%)", max: 80,
        bars: [OURS(52.9), Q25(26.8), Q3(52.7), Q35(65.2)]
      },
      {
        cat: "Mobile", title: "AndroidWorld", metric: "Task success rate (%)", max: 80,
        bars: [OURS(53.5), Q25(25.5), Q3(47.9), Q35(57.8)]
      },
      {
        cat: "Mobile", title: "MobileWorld", metric: "Task success rate (%)", max: 30,
        bars: [OURS(25.6), Q25(7.0), Q3(9.4), Q35(17.9)]
      },
      {
        cat: "Desktop", title: "OSWorld-Verified", metric: "Task success rate (%)", max: 50,
        foot: "The main remaining gap: very large screenshots with small targets expose high-resolution perception limits (see Analysis).",
        bars: [OURS(29.39), Q25(3.0), Q3(33.9), Q35(41.8)]
      },
      {
        cat: "Web", title: "WebVoyager", metric: "Task success rate (%)", max: 70,
        bars: [OURS(56.9), Q25(11.0), Q3(45.2), Q35(46.6)]
      }
    ],

    foundation: [
      {
        cat: "Reasoning", title: "MMMU (val)", metric: "Accuracy (%)", max: 80,
        bars: [OURS(56.2), LLADAV(48.6), SDAR(53.0), Q25(51.3), Q3(69.6)]
      },
      {
        cat: "Reasoning", title: "MathVista (mini)", metric: "Accuracy (%)", max: 100,
        bars: [OURS(70.5), LLADAV(59.7), SDAR(62.5), Q25(68.6), Q3(77.2)]
      },
      {
        cat: "Reasoning", title: "MathVerse (mini)", metric: "Accuracy (%)", max: 80,
        bars: [OURS(48.0), LLADAV(29.1), SDAR(36.6), Q25(49.2), Q3(62.1)]
      },
      {
        cat: "OCR & Chart", title: "ChartQA", metric: "Accuracy (%)", max: 100,
        bars: [OURS(84.8), LLADAV(82.7), SDAR(82.7), Q25(84.1), Q3(89.6)]
      },
      {
        cat: "OCR & Chart", title: "CharXiv (DQ)", metric: "Accuracy (%)", max: 100,
        bars: [OURS(72.2), LLADAV(47.0), SDAR(66.5), Q25(73.9), Q3(83.0)]
      },
      {
        cat: "OCR & Chart", title: "OCRBench", metric: "Score", max: 1000, digits: 0,
        bars: [OURS(855), LLADAV(632), SDAR(726), Q25(842), Q3(896)]
      }
    ],

    efficiency: [
      {
        cat: "Web input", title: "Web", metric: "Mean API latency (s) · ↓ lower is better", max: 20, digits: 2, suffix: " s",
        foot: "3.58× mean speedup (3.43× median) · LLaDA-UI generates 144 tokens vs 55.2 for Qwen3-VL-8B.",
        bars: [OURS(4.764), Q3(17.050)]
      },
      {
        cat: "Desktop input", title: "OSWorld", metric: "Mean API latency (s) · ↓ lower is better", max: 50, digits: 2, suffix: " s",
        foot: "6.83× mean speedup (6.78× median) · 129 vs 70 generated tokens.",
        bars: [OURS(6.379), Q3(43.545)]
      },
      {
        cat: "Mobile input", title: "MobileWorld", metric: "Mean API latency (s) · ↓ lower is better", max: 60, digits: 2, suffix: " s",
        foot: "8.95× mean speedup (8.81× median) · 67 vs 65.2 generated tokens.",
        bars: [OURS(5.921), Q3(53.000)]
      }
    ],

    analysis: [
      {
        cat: "AndroidWorld", title: "Success vs. optimal task length", metric: "Task success rate (%)", max: 80,
        foot: "116-task evaluation of an intermediate checkpoint · long trajectories degrade through repeated actions in unproductive UI states.",
        bars: [
          { model: "Tasks solvable in 1–5 optimal actions", name: "1–5", sub: "54 tasks", org: "34 successes", value: 62.96, kind: "ours" },
          { model: "Tasks solvable in 6–10 optimal actions", name: "6–10", sub: "35 tasks", org: "17 successes", value: 48.57, kind: "ours" },
          { model: "Tasks solvable in 11–20 optimal actions", name: "11–20", sub: "20 tasks", org: "8 successes", value: 40.0, kind: "ours" },
          { model: "Tasks needing more than 20 optimal actions", name: "20+", sub: "7 tasks", org: "1 success", value: 14.29, kind: "ours" }
        ]
      },
      {
        cat: "AndroidWorld", title: "Exact-repetition rate", metric: "Share of steps (%) · ↓ lower is better", max: 30,
        foot: "Repetition separates outcomes: failed trajectories repeat an identical action nearly five times more often.",
        bars: [
          { model: "Successful trajectories", name: "Successful", sub: "trajectories", org: "exact repeats", value: 5.41, kind: "ours" },
          { model: "Failed trajectories", name: "Failed", sub: "trajectories", org: "exact repeats", value: 26.7, kind: "diff" }
        ]
      }
    ]
  };

  window.LLADA_CHARTS = CHARTS;
})();
