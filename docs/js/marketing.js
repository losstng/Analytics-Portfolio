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

  businessImpact: `
    <div class="markdown">
      <h4>Key Business Impact</h4>
      <div class="metrics-grid">
        <div class="metric-card">
          <span class="metric-value">+105</span>
          <span class="metric-label">Incremental Customers/Month</span>
          <span class="metric-change">+4.2%</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">$1.07M</span>
          <span class="metric-label">Annual Revenue Impact</span>
          <span class="metric-change">+15% ROAS</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">$140</span>
          <span class="metric-label">Weighted CAC</span>
          <span class="metric-change">Under $150 Target</span>
        </div>
      </div>
    </div>
  `,

  executiveDashboard: `
    <figure>
      <figcaption>Executive Dashboard - Budget Optimization</figcaption>
      <iframe class="iframe-embed" style="height:700px" src="assets/marketing_analytics/executive_dashboard.html" loading="lazy" title="Executive Dashboard" aria-label="Executive dashboard showing conversion rates, CAC, budget allocation and impact"></iframe>
    </figure>
  `,

  dataQualityFramework: `
    <div class="quality-checks">
      <h4>Data Quality Framework</h4>
      <div class="check-grid">
        <div class="check-item pass">✅ Duplicate Detection: PASS (0 found)</div>
        <div class="check-item pass">✅ Spend Reconciliation: PASS (<2% variance)</div>
        <div class="check-item pass">✅ Join Integrity: PASS (100% valid)</div>
        <div class="check-item pass">✅ Null Audits: PASS (0 critical nulls)</div>
        <div class="check-item pass">✅ Anomaly Detection: NORMAL (all groups)</div>
      </div>
      <figure style="margin-top: 1rem;">
        <figcaption>Detailed Quality Report</figcaption>
        <iframe class="iframe-embed" style="height:300px" src="assets/marketing_analytics/quality_report.html" loading="lazy" title="Data Quality Report" aria-label="Detailed data quality report"></iframe>
      </figure>
    </div>
  `,

  scenarioAnalysis: `
    <div class="scenario-table">
      <h4>Scenario Modeling</h4>
      <figure>
        <figcaption>Budget Allocation Scenarios</figcaption>
        <iframe class="iframe-embed" style="height:280px" src="assets/marketing_analytics/scenarios_table.html" loading="lazy" title="Scenario Analysis" aria-label="Scenario analysis comparing conservative, recommended, and aggressive strategies"></iframe>
      </figure>
    </div>
  `,

  methodologyRigor: `
    <div class="rigor-scorecard">
      <h4>Methodological Rigor Score: 90/100</h4>
      <figure>
        <figcaption>Statistical Validation Framework</figcaption>
        <iframe class="iframe-embed" style="height:280px" src="assets/marketing_analytics/rigor_scorecard.html" loading="lazy" title="Methodology Rigor" aria-label="Methodological rigor scorecard showing validation checks"></iframe>
      </figure>
    </div>
  `,

  recommendations: `
    <div class="recommendations">
      <h4>Prioritized Recommendations</h4>
      <figure>
        <figcaption>Action Plan & Timeline</figcaption>
        <iframe class="iframe-embed" style="height:300px" src="assets/marketing_analytics/recommendations_timeline.html" loading="lazy" title="Recommendations" aria-label="Prioritized recommendations with timeline and impact"></iframe>
      </figure>
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
        <img src="assets/marketing_analytics/static_confidence_int.png" alt="Conversion rates with confidence intervals of A vs B vs C vs D" />
        <figcaption>Conversion rates with confidence intervals of A vs B vs C vs D</figcaption>
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
    link: '../Notebooks/marketing_analytics.ipynb',
    tags: ['Synthetic Data', 'RFM', 'A/B Testing'],
    contentHtml: marketingTemplates.overview
  },
  
  // Business Impact Summary - NEW
  {
    type: 'metrics',
    title: 'Business Impact Analysis',
    description: 'Projected impact of budget reallocation strategy with key KPIs.',
    tags: ['ROI', 'CAC', 'Revenue'],
    contentHtml: marketingTemplates.businessImpact
  },

  // Executive Dashboard - NEW
  {
    type: 'embed',
    title: 'Executive Dashboard',
    description: 'Interactive dashboard showing conversion rates, CAC analysis, and budget recommendations.',
    tags: ['Dashboard', 'Interactive', 'Plotly'],
    contentHtml: marketingTemplates.executiveDashboard
  },

  // Data Quality Framework - NEW
  {
    type: 'quality',
    title: 'Data Quality & Governance',
    description: 'Comprehensive data quality checks and governance framework.',
    tags: ['Data Quality', 'Governance'],
    contentHtml: marketingTemplates.dataQualityFramework
  },

  // Raw data sample
  {
    type: 'embed',
    title: 'Data Cleaning — Raw Sample',
    description: 'Preview of the raw data sample (5 rows) before cleaning.',
    tags: ['Data Quality', 'ETL'],
    contentHtml: marketingTemplates.rawDataSample
  },

  // Cleaned data sample
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
    title: 'A/B Testing — Tournament Results',
    description: 'Tournament-style A/B test results: D (11.4%) > B (8.3%) > A (4.8%) > C (3.5%)',
    tags: ['A/B Testing', 'Statistical Significance'],
    contentHtml: marketingTemplates.interactiveConversionRates
  },

  // Scenario Analysis - NEW
  {
    type: 'analysis',
    title: 'Scenario Modeling',
    description: 'Conservative vs Recommended vs Aggressive budget allocation scenarios.',
    tags: ['Scenario Analysis', 'Decision Support'],
    contentHtml: marketingTemplates.scenarioAnalysis
  },

  // Methodological Rigor - NEW
  {
    type: 'methodology',
    title: 'Methodological Rigor',
    description: 'Robustness checks and statistical validation framework.',
    tags: ['Statistics', 'Validation'],
    contentHtml: marketingTemplates.methodologyRigor
  },

  // Recommendations - NEW
  {
    type: 'recommendations',
    title: 'Action Plan & Timeline',
    description: 'Prioritized recommendations with effort estimates and projected payoff.',
    tags: ['Strategy', 'Implementation'],
    contentHtml: marketingTemplates.recommendations
  },

  // Static visuals
  {
    type: 'embed',
    title: 'Static Visuals — Summary',
    description: 'Publication-ready charts exported from the notebook.',
    tags: ['Seaborn', 'Reporting'],
    contentHtml: marketingTemplates.staticVisuals
  },

  // Summary KPI chart
  {
    type: 'chart',
    title: 'Customer Segments & Conversion Summary',
    description: 'Segment sizes and conversion rates by channel.',
    link: '../Notebooks/marketing_analytics.ipynb',
    tags: ['Segments', 'KPIs'],
    chartKey: 'Marketing'
  }
];

/**
 * Create chart configuration for Marketing category
 */
export function createMarketingChart(data) {
  // Use the new CAC analysis data if available
  if (data.marketing_cac_analysis) {
    const channels = Object.keys(data.marketing_cac_analysis);
    const conversionRates = channels.map(ch => data.marketing_cac_analysis[ch].conversion_rate);
    const cac = channels.map(ch => data.marketing_cac_analysis[ch].cac);
    
    return {
      type: 'bar',
      data: {
        labels: channels.map(ch => `Channel ${ch}`),
        datasets: [
          {
            label: 'Conversion Rate (%)',
            data: conversionRates.map(r => r * 100),
            backgroundColor: channels.map(ch => 
              ch === 'D' ? 'rgba(16, 185, 129, 0.8)' :  // Green for winner
              ch === 'C' ? 'rgba(239, 68, 68, 0.8)' :   // Red for underperformer
              'rgba(79, 70, 229, 0.6)'                   // Purple for others
            ),
            borderColor: channels.map(ch => 
              ch === 'D' ? 'rgba(16, 185, 129, 1)' :
              ch === 'C' ? 'rgba(239, 68, 68, 1)' :
              'rgba(79, 70, 229, 1)'
            ),
            borderWidth: 2,
            yAxisID: 'y',
          },
          {
            label: 'CAC ($)',
            data: cac,
            type: 'line',
            borderColor: 'rgba(251, 146, 60, 1)',
            backgroundColor: 'rgba(251, 146, 60, 0.1)',
            borderWidth: 2,
            tension: 0.3,
            yAxisID: 'y1',
            pointStyle: 'circle',
            pointRadius: 5,
            pointHoverRadius: 8
          }
        ],
      },
      options: {
        responsive: true,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: { position: 'top' },
          title: {
            display: true,
            text: 'Channel Performance: Conversion Rate vs CAC',
            font: { size: 14 }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                if (context.dataset.label === 'Conversion Rate (%)') {
                  return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                } else {
                  return `${context.dataset.label}: $${context.parsed.y.toFixed(0)}`;
                }
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Conversion Rate (%)' },
            max: 12
          },
          y1: {
            beginAtZero: true,
            position: 'right',
            title: { display: true, text: 'CAC ($)' },
            max: 5000,
            grid: { drawOnChartArea: false },
            ticks: {
              callback: (value) => `$${value}`
            }
          },
        },
      },
    };
  }
  
  // Fallback to original chart if new data not available
  const segmentLabels = Object.keys(data.marketing_segment_counts || {})
    .sort((a, b) => Number(a) - Number(b))
    .map((k) => `Segment ${k}`);
  const segmentVals = segmentLabels.map((lbl) => {
    const idx = lbl.split(' ')[1];
    return data.marketing_segment_counts[idx];
  });
  const convLabels = ['Group A', 'Group B'];
  const convVals = data.marketing_conversion_rates || [0.05, 0.08];

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
