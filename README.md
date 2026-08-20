# LLaDA-UI — Project Page

Static project page for **LLaDA-UI: An MoE-based Diffusion Vision-Language GUI Agent**
(Inclusion AI · Venus Team, Ant Group · Westlake University).

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
  denoise/                  # 75 recorded denoising frames (AndroidWorld case)
  traces/{web,os,mobile}/   # recorded agent trajectories
  framework.png             # architecture figure (from the report)
  data-pipeline.png         # GUI data pipeline figure (from the report)
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
- GitHub repository link (nav, hero, footer)
- Hugging Face model link (hero, footer)
