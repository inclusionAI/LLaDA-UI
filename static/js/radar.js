/* LLaDA-UI project page — overview radar, a native SVG re-rendering of the
 * report's hero figure. Each axis is independently normalized so the best
 * displayed raw score is 100%; raw maxima are printed in the axis labels. */

(function () {
  "use strict";

  var host = document.getElementById("radar-host");
  if (!host) return;

  var AXES = [
    { name: "ScreenSpot-V2", max: 94.0 },
    { name: "ScreenSpot-Pro", max: 65.2 },
    { name: "AndroidWorld", max: 57.8 },
    { name: "MobileWorld", max: 20.0 },
    { name: "OSWorld-V", max: 41.8 },
    { name: "WebVoyager", max: 55.9 }
  ];

  /* Raw benchmark scores in axis order. Qwen2.5-VL-7B's last three values are
     reported upper bounds and are plotted at the bounds. */
  var MODELS = [
    { name: "Qwen2.5-VL-7B", values: [86.0, 26.8, 25.5, 7.0, 3.0, 11.0], color: "#3d9a8b", dash: "2 4", width: 1.6, fill: 0 },
    { name: "Qwen3.5-9B", values: [94.0, 65.2, 57.8, 17.9, 41.8, 46.6], color: "#8a94a6", dash: "6 4", width: 1.6, fill: 0 },
    { name: "Qwen3-VL-8B", values: [93.0, 52.7, 47.9, 9.4, 33.9, 45.2], color: "#e07a5f", dash: "", width: 1.8, fill: 0.07 },
    { name: "LLaDA-UI", values: [90.7, 53.7, 51.2, 20.0, 15.0, 55.9], color: "#2a4b8c", dash: "", width: 2.6, fill: 0.15, dots: true }
  ];

  var NS = "http://www.w3.org/2000/svg";
  var W = 480, H = 428, CX = 240, CY = 212, R = 148;

  function pt(axis, frac) {
    var ang = (Math.PI / 2) - axis * (Math.PI / 3); // start top, clockwise
    return [CX + Math.cos(ang) * R * frac, CY - Math.sin(ang) * R * frac];
  }

  function make(tag, attrs) {
    var node = document.createElementNS(NS, tag);
    Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    return node;
  }

  var svg = make("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
    "aria-label": "Radar chart of GUI-agent benchmark results, each axis normalized to the best displayed score." });

  /* Grid rings */
  [20, 40, 60, 80, 100].forEach(function (level) {
    var d = "";
    for (var a = 0; a < 6; a++) {
      var p = pt(a, level / 100);
      d += (a ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1);
    }
    svg.appendChild(make("path", { d: d + "Z", fill: "none",
      stroke: level === 100 ? "#c3cde4" : "#dde3f1", "stroke-width": level === 100 ? 1.2 : 1 }));
  });

  /* Spokes */
  for (var a = 0; a < 6; a++) {
    var p = pt(a, 1);
    svg.appendChild(make("line", { x1: CX, y1: CY, x2: p[0], y2: p[1], stroke: "#dde3f1", "stroke-width": 1 }));
  }

  /* Ring labels along the top spoke */
  [20, 40, 60, 80, 100].forEach(function (level) {
    var p = pt(0, level / 100);
    var t = make("text", { x: p[0] + 5, y: p[1] + 3, "font-size": 8.5, fill: "#a7b1cc",
      "font-family": "IBM Plex Mono, monospace" });
    t.textContent = level + "%";
    svg.appendChild(t);
  });

  /* Model polygons */
  MODELS.forEach(function (m) {
    var points = m.values.map(function (v, i) {
      return pt(i, Math.min(1, v / AXES[i].max));
    });
    var d = points.map(function (p, i) {
      return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1);
    }).join("") + "Z";
    var attrs = { d: d, fill: m.fill ? m.color : "none", "fill-opacity": m.fill,
      stroke: m.color, "stroke-width": m.width, "stroke-linejoin": "round" };
    if (m.dash) attrs["stroke-dasharray"] = m.dash;
    svg.appendChild(make("path", attrs));
    if (m.dots) {
      points.forEach(function (p) {
        svg.appendChild(make("circle", { cx: p[0], cy: p[1], r: 3.1, fill: m.color, stroke: "#fff", "stroke-width": 1.2 }));
      });
    }
  });

  /* Axis labels: benchmark name + raw best score */
  AXES.forEach(function (axis, i) {
    var p = pt(i, 1.16);
    var anchor = "middle";
    if (i === 1 || i === 2) anchor = "start";
    if (i === 4 || i === 5) anchor = "end";
    var t = make("text", { x: p[0], y: p[1] - 4, "text-anchor": anchor, "font-size": 12,
      "font-weight": 600, fill: "#141d31", "font-family": "IBM Plex Sans, sans-serif" });
    t.textContent = axis.name;
    var sub = make("text", { x: p[0], y: p[1] + 10, "text-anchor": anchor, "font-size": 10,
      fill: "#58658a", "font-family": "IBM Plex Mono, monospace" });
    sub.textContent = "best " + axis.max.toFixed(1);
    svg.appendChild(t);
    svg.appendChild(sub);
  });

  host.appendChild(svg);
})();
