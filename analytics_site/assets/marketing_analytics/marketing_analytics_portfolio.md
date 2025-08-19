# Marketing Analytics — Notebook Summary and Web Assets

This document summarizes the `marketing_analytics.ipynb` workflow and enumerates the exported assets to embed on the portfolio website.

## Overview
- End-to-end synthetic marketing analytics: data generation, cleaning, segmentation, A/B testing, and visuals.
- Outputs include interactive tables/plots and static images prepared under `website/assets/marketing_analytics/`.

## Data Generation (Synthetic)
- Creates 5,000 customers with demographics, RFM features, device/channel, regions, dates, group assignment (A/B/C/D), and purchase outcome.
- Injects realistic issues: missing values, outliers (negative spend, impossible ages, income anomalies), and noisy categories.

## Cleaning Steps (Applied)
- Age: invalid (<18 or >100) -> NaN -> filled with median.
- Gender: normalized to first letter uppercase (M/F); unknowns remapped; filled with mode.
- Income: non-numeric and out-of-range (<1,000 or >200,000) -> NaN -> filled with median.
- Loyalty score: clipped to [0,1]; out-of-bounds -> NaN -> filled with median.
- Region / Preferred channel / Device: empty or unknown -> NaN -> filled with mode.
- Recency, Frequency, Monetary: negatives -> NaN -> filled with median.
- Account age: negatives -> NaN -> filled with median.
- Dates: signup forward-filled; last purchase filled with signup if missing.
- Group: filled with mode.
- Purchase: missing -> 0.

## Customer Segmentation
- KMeans with 2 clusters on RFM (`recency`, `frequency`, `monetary`).
- Segment labels added as `segment` for downstream analysis/visuals.

## A/B Testing (Tournament Style)
- Pairwise z-tests on conversion: A vs B, C vs D.
- Winners face off; losers face off; prints conversion rates, z-statistics, p-values, and ranked summary.

## Visualization Suite
- Conversion rates by group (seaborn bar).
- Pairwise comparison bars (A vs B, C vs D).
- Ranked bar chart by conversion rate.
- Conversion rates with 95% Wilson CIs.
- Tournament bracket sketch.
- Interactive conversion-rate bar chart (Plotly) for the web.

## Exported Assets (for Website)
All assets are saved to `website/assets/marketing_analytics/`:

- Data cleaning showcase:
  - `raw_data_sample.csv`, `raw_data_sample.html` — sample of raw rows (5).
  - `cleaned_data_sample.csv`, `cleaned_data_sample.html` — sample of cleaned rows (5).
  - `cleaned_data_table_interactive.html` — interactive HTML table of cleaned sample.
- Visuals:
  - `conversion_rates_interactive.html` — interactive Plotly bar chart of conversion rates.
  - `static_conversion_rates.png` — static conversion rate bar plot (cell 18 suite).
  - `static_pairwise_round1.png` — static A vs B and C vs D plots (cell 18 suite).

## How to Embed on the Website
In `website/projects/marketing_analytics.html`:

- Cleaning samples (tables):
  - Link to or include `assets/marketing_analytics/raw_data_sample.html` and `assets/marketing_analytics/cleaned_data_sample.html`.
  - Or embed the interactive table: `assets/marketing_analytics/cleaned_data_table_interactive.html` in an iframe.

- Interactive conversion bar chart:
  - Embed `assets/marketing_analytics/conversion_rates_interactive.html` in an iframe.

- Static images:
  - Reference `assets/marketing_analytics/static_conversion_rates.png` and `assets/marketing_analytics/static_pairwise_round1.png` with `<img>` tags.

Example snippet (iframes and images):

```html
<!-- Cleaning: Before vs After (interactive) -->
<iframe src="../assets/marketing_analytics/cleaned_data_table_interactive.html" width="100%" height="420" loading="lazy" style="border:0;"></iframe>

<!-- Interactive conversion rates -->
<iframe src="../assets/marketing_analytics/conversion_rates_interactive.html" width="100%" height="480" loading="lazy" style="border:0;"></iframe>

<!-- Static images -->
<img src="../assets/marketing_analytics/static_conversion_rates.png" alt="Conversion Rates by Group" style="max-width:100%;height:auto;" />
<img src="../assets/marketing_analytics/static_pairwise_round1.png" alt="Pairwise Conversion Rates (Round 1)" style="max-width:100%;height:auto;" />
```

Note: If `marketing_analytics.html` is in `website/projects/`, the relative path `../assets/marketing_analytics/...` is correct. Adjust if your file lives elsewhere.

## Section Structure to Reuse
- Introduction and objectives
- Data generation (synthetic) overview
- Data cleaning steps (bulleted) + before/after samples
- Segmentation results (brief) and what segments imply
- A/B testing tournament (what was tested and why)
- Visualizations (interactive + static) and insights
- Key takeaways

## Next Steps
- Add short narrative captions under each asset on the project page.
- Optionally, publish full cleaned dataset or a larger sample for download.
- Add a brief “Methods”/“Assumptions” note for statistical tests.
