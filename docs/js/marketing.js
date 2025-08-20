/**
 * Marketing analytics domain module.
 * Contains templates, chart configuration, and data specific to marketing projects.
 */

import { DASHBOARD_DATA } from './data.js';

/**
 * HTML templates for marketing project content
 */
export const marketingTemplates = {
  overview: `
    <div class="markdown">
      <p>This project generates a 5,000-customer synthetic dataset; cleans missing, out-of-range, and noisy values; segments customers via KMeans on RFM; and runs a tournament-style A/B test across groups A–D.</p>
      <h4>Cleaning steps (applied)</h4>
      <ul>
        <li><strong>Age</strong>: invalid (&lt;18 or &gt;100) ➜ NaN ➜ median fill</li>
        <li><strong>Gender</strong>: normalized to M/F; unknowns remapped; mode fill</li>
        <li><strong>Income</strong>: non-numeric/out-of-range (&lt;1k or &gt;200k) ➜ NaN ➜ median fill</li>
        <li><strong>Loyalty</strong>: clipped to [0,1]; out-of-bounds ➜ NaN ➜ median fill</li>
        <li><strong>Region/Channel/Device</strong>: empty/unknown ➜ NaN ➜ mode fill</li>
        <li><strong>R/F/M</strong>: negatives ➜ NaN ➜ median fill</li>
        <li><strong>Account age</strong>: negatives ➜ NaN ➜ median fill</li>
        <li><strong>Dates</strong>: signup ffill; last purchase ➜ signup if missing</li>
        <li><strong>Group</strong>: mode fill; <strong>Purchase</strong>: missing ➜ 0</li>
      </ul>
    </div>
  `,

  rawDataSample: `
    <figure>
      <figcaption>Raw sample (5 rows)</figcaption>
      <iframe class="iframe-embed iframe-csv" src="assets/marketing_analytics/raw_data_sample.html" loading="lazy" title="Raw data sample" aria-label="Raw data sample"></iframe>
      <div class="downloads"><a class="btn" href="assets/marketing_analytics/raw_data_sample.csv" download>Download CSV</a></div>
    </figure>
  `,

  cleanedDataSample: `
    <figure>
      <figcaption>Cleaned sample (5 rows)</figcaption>
      <iframe class="iframe-embed iframe-csv" src="assets/marketing_analytics/cleaned_data_sample.html" loading="lazy" title="Cleaned data sample" aria-label="Cleaned data sample"></iframe>
      <div class="downloads"><a class="btn" href="assets/marketing_analytics/cleaned_data_sample.csv" download>Download CSV</a></div>
    </figure>
    <details>
      <summary>Interactive table</summary>
      <iframe class="iframe-embed" style="height:420px" src="assets/marketing_analytics/cleaned_data_table_interactive.html" loading="lazy" title="Interactive cleaned table" aria-label="Interactive cleaned table"></iframe>
    </details>
  `,

  interactiveConversionRates: `
    <iframe class="iframe-embed" style="height:420px" src="assets/marketing_analytics/conversion_rates_interactive.html" loading="lazy" title="Interactive conversion rates" aria-label="Interactive conversion rates"></iframe>
  `,

  staticVisuals: `
    <div class="image-grid">
      <figure>
        <img src="assets/marketing_analytics/static_conversion_rates.png" alt="Conversion rates by group" />
        <figcaption>Conversion rates by group (with labels)</figcaption>
      </figure>
      <figure>
        <img src="assets/marketing_analytics/static_pairwise_round1.png" alt="Pairwise conversion rates A vs B, C vs D" />
        <figcaption>Round 1 pairwise comparisons (A vs B, C vs D)</figcaption>
      </figure>
    </div>
  `
};

/**
 * Marketing project configuration
 */
export const marketingProjects = [
  // Overview & narrative
  {
    type: 'markdown',
    title: 'Marketing Analytics — Overview',
    description: 'End-to-end synthetic marketing analytics: generation, cleaning, segmentation, and A/B testing.',
    link: '../Scripts/marketing_analytics.py',
    tags: ['Synthetic Data', 'RFM', 'A/B Testing'],
    contentHtml: marketingTemplates.overview
  },
  // Raw data sample (CSV preview)
  {
    type: 'embed',
    title: 'Data Cleaning — Raw Sample',
    description: 'Preview of the raw data sample (5 rows) before cleaning.',
    tags: ['Data Quality', 'ETL'],
    contentHtml: marketingTemplates.rawDataSample
  },
  // Cleaned data sample (CSV preview)
  {
    type: 'embed',
    title: 'Data Cleaning — Cleaned Sample',
    description: 'Preview of the cleaned data sample (5 rows) after cleaning.',
    tags: ['Data Quality', 'ETL'],
    contentHtml: marketingTemplates.cleanedDataSample
  },
  // Interactive conversion chart
  {
    type: 'embed',
    title: 'Interactive — Conversion Rates by Group',
    description: 'Explore conversion rates across groups (A/B/C/D) with hover details.',
    tags: ['Plotly', 'Interactivity'],
    contentHtml: marketingTemplates.interactiveConversionRates
  },
  // Static visuals (exported from notebook)
  {
    type: 'embed',
    title: 'Static Visuals — Summary',
    description: 'Publication-ready charts exported from the notebook.',
    tags: ['Seaborn', 'Reporting'],
    contentHtml: marketingTemplates.staticVisuals
  },
  // Summary KPI chart using embedded DASHBOARD_DATA
  {
    type: 'chart',
    title: 'Customer Segments & A/B Overview',
    description: 'Segment sizes and A/B conversion rates summary.',
    link: '../Scripts/marketing_analytics.py',
    tags: ['Segments', 'KPIs'],
    chartKey: 'Marketing'
  }
];

/**
 * Create chart configuration for Marketing category
 */
export function createMarketingChart(data) {
  // Segment counts and conversion rates: two datasets on different axes
  const segmentLabels = Object.keys(data.marketing_segment_counts)
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => `Segment ${k}`);
  const segmentVals = segmentLabels.map((lbl) => {
    const idx = lbl.split(' ')[1];
    return data.marketing_segment_counts[idx];
  });
  const convLabels = ['Group A', 'Group B'];
  const convVals = data.marketing_conversion_rates;

  // Combine labels: segments first, then groups
  const labels = [...segmentLabels, ...convLabels];
  const countsExtended = [
    ...segmentVals,
    null,
    null,
  ];
  const ratesExtended = [
    null,
    null,
    null,
    convVals[0],
    convVals[1],
  ];

  return {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Segment Size',
          data: countsExtended,
          backgroundColor: 'rgba(79, 70, 229, 0.6)',
          borderColor: 'rgba(79, 70, 229, 1)',
          borderWidth: 1,
          yAxisID: 'y',
        },
        {
          label: 'Conversion Rate',
          data: ratesExtended,
          backgroundColor: 'rgba(16, 185, 129, 0.6)',
          borderColor: 'rgba(16, 185, 129, 1)',
          borderWidth: 1,
          yAxisID: 'y1',
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
        title: {
          display: true,
          text: 'Customer Segments & Conversion Rates',
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: 'Segment Size' },
        },
        y1: {
          beginAtZero: true,
          position: 'right',
          title: { display: true, text: 'Conversion Rate' },
          ticks: {
            callback: (value) => `${(value * 100).toFixed(1)}%`,
          },
          grid: {
            drawOnChartArea: false,
          },
        },
      },
    },
  };
}
