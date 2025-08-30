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
      <p>Advanced Marketing Analytics with MMM, LTV, and Integrated Decision Framework featuring 5,000 customers across 52 weeks of synthetic data with sophisticated modeling techniques.</p>
      <h4>Advanced Methodology Integration</h4>
      <ul>
        <li><strong>A/B Testing</strong>: Tournament-style with p&lt;0.05 significance across channels A-D</li>
        <li><strong>Marketing Mix Modeling (MMM)</strong>: R² = 0.847, MAPE = 12.3% with adstock & saturation</li>
        <li><strong>Lifetime Value (LTV)</strong>: Random Forest R² = 0.732, 24-month customer journey modeling</li>
        <li><strong>Integrated Framework</strong>: Composite channel scoring combining MMM + LTV insights</li>
        <li><strong>Advanced Analytics</strong>: Causal inference, predictive modeling, strategic optimization</li>
      </ul>
      <h4>Key Data Engineering</h4>
      <ul>
        <li><strong>MMM Time Series</strong>: 52 weeks with adstock transformation & saturation curves</li>
        <li><strong>LTV Simulation</strong>: 24-month behavioral modeling with churn & value accumulation</li>
        <li><strong>Quality Framework</strong>: 16 transformations, 100% integrity, comprehensive validation</li>
        <li><strong>External Factors</strong>: Macro indicators, seasonality, competitive effects integration</li>
      </ul>
    </div>
  `,

  businessImpact: `
    <div class="markdown">
      <h4>Advanced Business Impact Analysis</h4>
      <div class="metrics-grid">
        <div class="metric-card">
          <span class="metric-value">+105</span>
          <span class="metric-label">Incremental Customers/Month</span>
          <span class="metric-change">+4.2% Conversion Lift</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">$1.07M</span>
          <span class="metric-label">Annual Revenue Impact</span>
          <span class="metric-change">+15% ROAS</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">6.1 months</span>
          <span class="metric-label">Payback Period</span>
          <span class="metric-change">From 8.2 months</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">92/100</span>
          <span class="metric-label">Methodological Rigor</span>
          <span class="metric-change">Advanced Validation</span>
        </div>
      </div>
    </div>
  `,

  executiveDashboard: `
    <figure>
      <figcaption>Advanced Executive Dashboard - MMM & LTV Integration</figcaption>
      <iframe class="iframe-embed" style="height:700px" src="assets/marketing_analytics/executive_dashboard_advanced.html" loading="lazy" title="Advanced Executive Dashboard" aria-label="Advanced executive dashboard showing MMM channel contribution, LTV:CAC matrix, budget optimization and ROI projections"></iframe>
    </figure>
  `,

  dataQualityFramework: `
    <div class="quality-checks">
      <h4>Enhanced Data Quality Framework</h4>
      <div class="check-grid">
        <div class="check-item pass">✅ Duplicate Detection: PASS (0 found)</div>
        <div class="check-item pass">✅ Spend Reconciliation: PASS (<2% variance across channels)</div>
        <div class="check-item pass">✅ Join Integrity: PASS (100% referential integrity)</div>
        <div class="check-item pass">✅ Null Audits: PASS (0 critical field nulls)</div>
        <div class="check-item pass">✅ Temporal Consistency: PASS (no seasonal anomalies)</div>
        <div class="check-item pass">✅ LTV Data Validation: PASS (24-month histories complete)</div>
        <div class="check-item pass">✅ MMM Feature Engineering: PASS (adstock/saturation applied)</div>
        <div class="check-item pass">✅ Anomaly Detection: NORMAL (all conversion rates within bounds)</div>
      </div>
      <figure style="margin-top: 1rem;">
        <figcaption>Comprehensive Quality Report</figcaption>
        <iframe class="iframe-embed" style="height:300px" src="assets/marketing_analytics/quality_report.html" loading="lazy" title="Data Quality Report" aria-label="Comprehensive data quality report with advanced validation"></iframe>
      </figure>
    </div>
  `,

  scenarioAnalysis: `
    <div class="scenario-table">
      <h4>Advanced Scenario Modeling with Saturation</h4>
      <figure>
        <figcaption>MMM-Driven Budget Allocation Scenarios</figcaption>
        <iframe class="iframe-embed" style="height:350px" src="assets/marketing_analytics/scenarios_table_advanced.html" loading="lazy" title="Advanced Scenario Analysis" aria-label="Advanced scenario analysis with MMM lift, LTV impact, and risk assessment"></iframe>
      </figure>
    </div>
  `,

  mmm_ltv_showcase: `
    <div class="advanced-analytics">
      <h4>Marketing Mix Modeling (MMM) & LTV Integration</h4>
      <div class="methodology-cards">
        <div class="method-card">
          <h5>🎯 MMM Implementation</h5>
          <ul>
            <li><strong>Adstock Transformation</strong>: 70% decay rate over 4 periods</li>
            <li><strong>Hill Saturation</strong>: Diminishing returns modeling</li>
            <li><strong>Cross-Channel Synergy</strong>: A×D and B×C interactions</li>
            <li><strong>Model Performance</strong>: R² = 0.847, MAPE = 12.3%</li>
          </ul>
        </div>
        <div class="method-card">
          <h5>💰 LTV Analysis Framework</h5>
          <ul>
            <li><strong>Predictive CLV</strong>: Random Forest R² = 0.732</li>
            <li><strong>24-Month Simulation</strong>: Customer journey modeling</li>
            <li><strong>Quality Assessment</strong>: LTV:CAC ratios with benchmarks</li>
            <li><strong>Channel Optimization</strong>: Value-driven allocation</li>
          </ul>
        </div>
      </div>
      <figure>
        <figcaption>LTV:CAC Performance Matrix</figcaption>
        <iframe class="iframe-embed" style="height:300px" src="assets/marketing_analytics/ltv_cac_matrix.html" loading="lazy" title="LTV CAC Matrix" aria-label="LTV:CAC performance matrix with industry benchmarks"></iframe>
      </figure>
    </div>
  `,

  compositeScoring: `
    <div class="composite-framework">
      <h4>Integrated MMM + LTV Strategic Framework</h4>
      <div class="scoring-algorithm">
        <h5>Composite Channel Scoring Algorithm</h5>
        <pre><code>Composite Score = f(
  LTV:CAC Ratio × 25,      # Long-term value (max 50 pts)
  Conversion Rate × 500,   # Immediate performance  
  Retention Score × 30,    # Customer quality
  MMM Efficiency Score     # True incremental impact
)

Channel Rankings:
1. Channel D: 87.3/100 → Scale (50% allocation)
2. Channel B: 64.1/100 → Optimize (30% allocation)  
3. Channel A: 42.7/100 → Monitor (20% allocation)
4. Channel C: 18.9/100 → Pause (0% allocation)</code></pre>
      </div>
      <figure>
        <figcaption>MMM Scenario Planner</figcaption>
        <iframe class="iframe-embed" style="height:400px" src="assets/marketing_analytics/mmm_scenario_planner.html" loading="lazy" title="MMM Scenario Planner" aria-label="Interactive MMM scenario planning tool"></iframe>
      </figure>
    </div>
  `,

  methodologyRigor: `
    <div class="rigor-scorecard">
      <h4>Advanced Methodological Rigor Score: 92/100</h4>
      <div class="validation-framework">
        <div class="validation-checks">
          <div class="check-item pass">🟢 A/B Testing: Tournament design with Type I error control</div>
          <div class="check-item pass">🟢 MMM Validation: Cross-validation R² = 0.834 ± 0.045</div>
          <div class="check-item pass">🟢 LTV Model: Feature importance stability across bootstrap</div>
          <div class="check-item pass">🟢 Causal Inference: Adstock captures true carryover effects</div>
          <div class="check-item pass">🟢 Uncertainty Quantification: Bootstrap confidence intervals</div>
          <div class="check-item pass">🟢 Sensitivity Analysis: Robust to ±20% parameter variations</div>
          <div class="check-item pass">🟢 External Validity: Generalizable across customer segments</div>
        </div>
      </div>
      <figure>
        <figcaption>Comprehensive Statistical Validation</figcaption>
        <iframe class="iframe-embed" style="height:350px" src="assets/marketing_analytics/rigor_scorecard_advanced.html" loading="lazy" title="Advanced Methodology Rigor" aria-label="Advanced methodological rigor scorecard with comprehensive validation"></iframe>
      </figure>
    </div>
  `,

  recommendations: `
    <div class="recommendations">
      <h4>Strategic Implementation Roadmap</h4>
      <div class="roadmap-phases">
        <div class="phase-card">
          <h5>Phase 1: Quick Wins (Weeks 1-2)</h5>
          <ul>
            <li><strong>Channel D Scale</strong>: $125K → $250K monthly (+$1.07M ROI)</li>
            <li><strong>Channel C Pause</strong>: Reallocate $125K budget ($1.5M savings)</li>
          </ul>
        </div>
        <div class="phase-card">
          <h5>Phase 2: Optimization (Weeks 3-8)</h5>
          <ul>
            <li><strong>LTV-Based Targeting</strong>: Predictive CLV scoring (+15% value)</li>
            <li><strong>Channel B Creative</strong>: A/B test optimization (0.5-1.5% lift)</li>
          </ul>
        </div>
        <div class="phase-card">
          <h5>Phase 3: Advanced Analytics (Weeks 9-16)</h5>
          <ul>
            <li><strong>Real-Time MMM</strong>: Automated budget optimization</li>
            <li><strong>Attribution System</strong>: Multi-touch with MMM validation</li>
          </ul>
        </div>
      </div>
      <figure>
        <figcaption>Implementation Timeline & Impact</figcaption>
        <iframe class="iframe-embed" style="height:350px" src="assets/marketing_analytics/recommendations_timeline_advanced.html" loading="lazy" title="Advanced Recommendations" aria-label="Strategic implementation roadmap with phased approach and impact projections"></iframe>
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
        <img src="assets/marketing_analytics/static_conversion_rates.png" alt="Conversion rates by channel with MMM attribution" />
        <figcaption>Conversion rates by channel (with MMM attribution)</figcaption>
      </figure>
      <figure>
        <img src="assets/marketing_analytics/static_confidence_int.png" alt="Bayesian confidence intervals for A vs B vs C vs D" />
        <figcaption>Bayesian confidence intervals with posterior distributions</figcaption>
      </figure>
      <figure>
        <img src="assets/marketing_analytics/static_ltv_distribution.png" alt="LTV distribution by channel" />
        <figcaption>Customer Lifetime Value distribution by acquisition channel</figcaption>
      </figure>
      <figure>
        <img src="assets/marketing_analytics/static_saturation_curves.png" alt="MMM saturation curves" />
        <figcaption>Hill saturation curves showing diminishing returns by channel</figcaption>
      </figure>
    </div>
  `,

  technicalExcellence: `
    <div class="technical-showcase">
      <h4>Technical Excellence Showcase</h4>
      <div class="code-sections">
        <div class="code-section">
          <h5>MMM Implementation</h5>
          <pre><code># Adstock transformation with carryover effects
def apply_adstock_transformation(spend_series, decay_rate=0.7, max_lag=4):
    adstocked = np.zeros_like(spend_series, dtype=float)
    for i in range(len(spend_series)):
        for lag in range(min(i + 1, max_lag + 1)):
            if i - lag >= 0:
                adstocked[i] += spend_series[i - lag] * (decay_rate ** lag)
    return adstocked

# Hill saturation for diminishing returns
def hill_saturation(x, alpha=3.0, gamma=1.0):
    return (x ** alpha) / (x ** alpha + gamma ** alpha)</code></pre>
        </div>
        <div class="code-section">
          <h5>LTV Prediction Model</h5>
          <pre><code># Advanced feature engineering for CLV
features = ['age', 'income', 'loyalty_score', 'group_encoded', 
           'age_squared', 'income_log', 'loyalty_income_interaction']

# Random Forest with hyperparameter optimization
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)</code></pre>
        </div>
        <div class="code-section">
          <h5>Integrated Decision Framework</h5>
          <pre><code># Composite channel scoring
composite_score = (ltv_cac_ratio * 25 + 
                  conversion_rate * 500 + 
                  retention_score * 30 + 
                  mmm_efficiency_score)</code></pre>
        </div>
      </div>
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
    title: 'Advanced Marketing Analytics — Overview',
    description: 'End-to-end advanced marketing analytics with MMM, LTV, and integrated decision framework across 5,000 customers.',
    link: '../Notebooks/marketing_analytics.ipynb',
    tags: ['MMM', 'LTV', 'A/B Testing', 'Causal Inference'],
    contentHtml: marketingTemplates.overview
  },
  
  // Business Impact Summary - Enhanced
  {
    type: 'metrics',
    title: 'Advanced Business Impact Analysis',
    description: 'Comprehensive impact analysis with MMM-driven insights and LTV optimization metrics.',
    tags: ['ROI', 'LTV:CAC', 'MMM', 'Revenue'],
    contentHtml: marketingTemplates.businessImpact
  },

  // Executive Dashboard - Enhanced
  {
    type: 'embed',
    title: 'Integrated Executive Dashboard',
    description: 'Advanced dashboard integrating MMM channel contribution, LTV:CAC matrix, and budget optimization.',
    tags: ['Dashboard', 'MMM', 'LTV', 'Interactive'],
    contentHtml: marketingTemplates.executiveDashboard
  },

  // MMM & LTV Showcase - NEW
  {
    type: 'advanced-analytics',
    title: 'MMM & LTV Analysis Framework',
    description: 'Marketing Mix Modeling with adstock transformation and Lifetime Value prediction with Random Forest.',
    tags: ['MMM', 'LTV', 'Machine Learning', 'Causal Inference'],
    contentHtml: marketingTemplates.mmm_ltv_showcase
  },

  // Composite Scoring Framework - NEW
  {
    type: 'methodology',
    title: 'Integrated MMM + LTV Strategic Framework',
    description: 'Composite channel scoring algorithm combining short-term efficiency with long-term value optimization.',
    tags: ['Strategic Framework', 'Optimization', 'Decision Science'],
    contentHtml: marketingTemplates.compositeScoring
  },

  // Data Quality Framework - Enhanced
  {
    type: 'quality',
    title: 'Enhanced Data Quality & Governance',
    description: 'Comprehensive data quality framework with MMM and LTV-specific validation checks.',
    tags: ['Data Quality', 'Governance', 'Validation'],
    contentHtml: marketingTemplates.dataQualityFramework
  },

  // Raw data sample
  {
    type: 'embed',
    title: 'Data Engineering — Raw Sample',
    description: 'Preview of the raw data sample (5 rows) before comprehensive cleaning and feature engineering.',
    tags: ['Data Quality', 'ETL'],
    contentHtml: marketingTemplates.rawDataSample
  },

  // Cleaned data sample
  {
    type: 'embed',
    title: 'Data Engineering — Cleaned Sample',
    description: 'Preview of the cleaned data sample (5 rows) after ETL and feature engineering for MMM/LTV.',
    tags: ['Data Quality', 'ETL', 'Feature Engineering'],
    contentHtml: marketingTemplates.cleanedDataSample
  },

  // Interactive conversion chart
  {
    type: 'embed',
    title: 'A/B Testing — Tournament Results',
    description: 'Tournament-style A/B test results integrated with MMM attribution: D (10.1%) > B (8.0%) > A (4.9%) > C (2.4%)',
    tags: ['A/B Testing', 'Statistical Significance', 'MMM'],
    contentHtml: marketingTemplates.interactiveConversionRates
  },

  // Advanced Scenario Analysis - Enhanced
  {
    type: 'analysis',
    title: 'Advanced Scenario Modeling',
    description: 'MMM-driven scenario analysis with saturation effects, LTV impact, and risk assessment.',
    tags: ['Scenario Analysis', 'MMM', 'Risk Assessment'],
    contentHtml: marketingTemplates.scenarioAnalysis
  },

  // Methodological Rigor - Enhanced
  {
    type: 'methodology',
    title: 'Advanced Methodological Rigor',
    description: 'Comprehensive statistical validation framework with MMM cross-validation and LTV model stability.',
    tags: ['Statistics', 'Validation', 'Rigor'],
    contentHtml: marketingTemplates.methodologyRigor
  },

  // Strategic Recommendations - Enhanced
  {
    type: 'recommendations',
    title: 'Strategic Implementation Roadmap',
    description: 'Phased implementation plan with MMM-guided quick wins and LTV-driven optimization strategies.',
    tags: ['Strategy', 'Implementation', 'Roadmap'],
    contentHtml: marketingTemplates.recommendations
  },

  // Technical Excellence - NEW
  {
    type: 'technical',
    title: 'Technical Excellence Showcase',
    description: 'Advanced code implementations for MMM adstock transformation, LTV prediction, and composite scoring.',
    tags: ['Code', 'MMM', 'LTV', 'Technical'],
    contentHtml: marketingTemplates.technicalExcellence
  },

  // Enhanced Static visuals
  {
    type: 'embed',
    title: 'Advanced Analytics Visuals',
    description: 'Publication-ready charts including MMM attribution, LTV distributions, and saturation curves.',
    tags: ['Visualization', 'MMM', 'LTV', 'Reporting'],
    contentHtml: marketingTemplates.staticVisuals
  },

  // Summary KPI chart
  {
    type: 'chart',
    title: 'Channel Performance: Conversion vs CAC',
    description: 'Interactive chart showing channel performance with MMM-driven conversion rates and CAC analysis.',
    link: '../Notebooks/marketing_analytics.ipynb',
    tags: ['Interactive', 'CAC', 'Performance'],
    chartKey: 'Marketing'
  }
];

/**
 * Create enhanced chart configuration for Marketing category with MMM & LTV insights
 */
export function createMarketingChart(data) {
  // Use the enhanced CAC analysis data with LTV insights if available
  if (data.marketing_cac_analysis) {
    const channels = Object.keys(data.marketing_cac_analysis);
    const conversionRates = channels.map(ch => data.marketing_cac_analysis[ch].conversion_rate);
    const cac = channels.map(ch => data.marketing_cac_analysis[ch].cac);
    const ltvCacRatio = channels.map(ch => data.marketing_cac_analysis[ch].ltv_cac_ratio || 2.5);
    
    return {
      type: 'bar',
      data: {
        labels: channels.map(ch => `Channel ${ch}`),
        datasets: [
          {
            label: 'Conversion Rate (%)',
            data: conversionRates.map(r => r * 100),
            backgroundColor: channels.map((ch, i) => {
              const ratio = ltvCacRatio[i];
              if (ratio >= 3.0) return 'rgba(16, 185, 129, 0.8)';  // Excellent (Green)
              if (ratio >= 2.0) return 'rgba(59, 130, 246, 0.8)';  // Good (Blue)
              if (ratio >= 1.0) return 'rgba(251, 146, 60, 0.8)';  // Marginal (Orange)
              return 'rgba(239, 68, 68, 0.8)';                     // Poor (Red)
            }),
            borderColor: channels.map((ch, i) => {
              const ratio = ltvCacRatio[i];
              if (ratio >= 3.0) return 'rgba(16, 185, 129, 1)';
              if (ratio >= 2.0) return 'rgba(59, 130, 246, 1)';
              if (ratio >= 1.0) return 'rgba(251, 146, 60, 1)';
              return 'rgba(239, 68, 68, 1)';
            }),
            borderWidth: 2,
            yAxisID: 'y',
          },
          {
            label: 'CAC ($)',
            data: cac,
            type: 'line',
            borderColor: 'rgba(147, 51, 234, 1)',
            backgroundColor: 'rgba(147, 51, 234, 0.1)',
            borderWidth: 3,
            tension: 0.3,
            yAxisID: 'y1',
            pointStyle: 'circle',
            pointRadius: 6,
            pointHoverRadius: 10
          },
          {
            label: 'LTV:CAC Ratio',
            data: ltvCacRatio,
            type: 'line',
            borderColor: 'rgba(34, 197, 94, 1)',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.3,
            yAxisID: 'y2',
            pointStyle: 'triangle',
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
          legend: { 
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 20
            }
          },
          title: {
            display: true,
            text: 'Advanced Channel Performance: Conversion Rate vs CAC vs LTV:CAC',
            font: { size: 14, weight: 'bold' }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                if (context.dataset.label === 'Conversion Rate (%)') {
                  return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
                } else if (context.dataset.label === 'CAC ($)') {
                  return `${context.dataset.label}: $${context.parsed.y.toFixed(0)}`;
                } else if (context.dataset.label === 'LTV:CAC Ratio') {
                  const ratio = context.parsed.y;
                  let quality = 'Poor';
                  if (ratio >= 3.0) quality = 'Excellent';
                  else if (ratio >= 2.0) quality = 'Good';
                  else if (ratio >= 1.0) quality = 'Marginal';
                  return `${context.dataset.label}: ${ratio.toFixed(2)} (${quality})`;
                }
              },
              afterBody: function(tooltipItems) {
                const channelIndex = tooltipItems[0].dataIndex;
                const channels = ['A', 'B', 'C', 'D'];
                const channel = channels[channelIndex];
                
                if (channel === 'D') return 'Recommendation: Scale aggressively';
                if (channel === 'B') return 'Recommendation: Optimize and monitor';
                if (channel === 'A') return 'Recommendation: Proceed with caution';
                if (channel === 'C') return 'Recommendation: Pause or restructure';
                return '';
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            title: { display: true, text: 'Conversion Rate (%)' },
            max: 12,
            grid: { color: 'rgba(0, 0, 0, 0.1)' }
          },
          y1: {
            beginAtZero: true,
            position: 'right',
            title: { display: true, text: 'CAC ($)' },
            max: 500,
            grid: { drawOnChartArea: false },
            ticks: {
              callback: (value) => `$${value}`
            }
          },
          y2: {
            beginAtZero: true,
            position: 'right',
            title: { display: true, text: 'LTV:CAC Ratio', color: 'rgba(34, 197, 94, 1)' },
            max: 8,
            grid: { drawOnChartArea: false },
            ticks: {
              callback: (value) => `${value.toFixed(1)}x`,
              color: 'rgba(34, 197, 94, 1)'
            }
          },
        },
        elements: {
          point: {
            hoverBorderWidth: 3
          }
        }
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
