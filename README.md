# LLaDA-UI — Project Page

Static project page for **LLaDA-UI: Bringing Block-wise Diffusion to Vision-Language GUI Agents**
(AGI Research Center, Inclusion AI · Venus Team, Ant Group · Westlake University).
All numbers and claims mirror the final technical report.

## Structure

```
index.html                  # the whole page
static/css/                 # base tokens, layout, components, charts, players
static/js/
  results-data.js           # all chart numbers (edit here to update results)
  charts.js                 # bar-chart renderer, tabs, tooltip, lazy tables
  radar.js                  # overview radar (SVG, per-axis normalized)
  player.js                 # frame-sequence players (denoising + traces)
  main.js                   # denoising headline, reveal, count-up, BibTeX copy
partials/tables/            # full comparison tables, lazy-loaded on expand
static/images/
  denoise/                  # 75 recorded denoising frames (AndroidWorld hero case)
  denoise-cases/            # 132 frames: two more complete decoding cases (Type, Click)
  traces/{web,os,mobile}/   # recorded agent trajectories
  framework.png             # architecture figure (from the report)
  data-pipeline.png         # GUI data pipeline figure (from the report)
  case10-input.png          # static observation for the hero denoising case
  inclusion-antgroup.png    # corporate logos (footer); llada-ui-logo.png is an unused hi-res duplicate
  favicon.png
```

## Local preview

Table partials are fetched at runtime, so open the page through a server:

```bash
python3 -m http.server 8000
# → http://localhost:8000
```

## Pending TODOs

Search the sources for `TODO` / `SOON`:

- arXiv link + identifier (hero button, BibTeX, footer)

The GitHub link (https://github.com/inclusionAI/LLaDA-UI) is live in the nav, hero, and footer;
the Hugging Face link (https://huggingface.co/inclusionAI/LLaDA-UI) is live in the hero and footer.
