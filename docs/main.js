// Data model for projects. Each entry includes a title, description,
// optional link to the underlying Python script, tags, and a type:
// - 'chart'  -> renders a Chart.js canvas using precomputed data
// - 'embed'  -> renders provided HTML (iframes, images, etc.)
// - 'markdown' -> renders provided HTML text content
// When omitted, type defaults to 'chart' for backward compatibility.
const projects = {
  Operations: [
    {
      title: 'Supply Chain Optimization',
      description: 'Optimize inventory using forecasting and EOQ.',
      link: '../Scripts/supply_chain_optimization.py',
      tags: ['Forecasting', 'EOQ'],
      chartKey: 'Operations'
    }
  ],
  Marketing: [
    // Overview & narrative
    {
      type: 'markdown',
      title: 'Marketing Analytics — Overview',
      description: 'End-to-end synthetic marketing analytics: generation, cleaning, segmentation, and A/B testing.',
      link: '../Scripts/marketing_analytics.py',
      tags: ['Synthetic Data', 'RFM', 'A/B Testing'],
      contentHtml: `
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
      `
    },
    // Raw data sample (CSV preview)
    {
      type: 'embed',
      title: 'Data Cleaning — Raw Sample',
      description: 'Preview of the raw data sample (5 rows) before cleaning.',
      tags: ['Data Quality', 'ETL'],
      contentHtml: `
        <figure>
          <figcaption>Raw sample (5 rows)</figcaption>
          <iframe class="iframe-embed iframe-csv" src="assets/marketing_analytics/raw_data_sample.html" loading="lazy" title="Raw data sample" aria-label="Raw data sample"></iframe>
          <div class="downloads"><a class="btn" href="assets/marketing_analytics/raw_data_sample.csv" download>Download CSV</a></div>
        </figure>
      `
    },
    // Cleaned data sample (CSV preview)
    {
      type: 'embed',
      title: 'Data Cleaning — Cleaned Sample',
      description: 'Preview of the cleaned data sample (5 rows) after cleaning.',
      tags: ['Data Quality', 'ETL'],
      contentHtml: `
        <figure>
          <figcaption>Cleaned sample (5 rows)</figcaption>
          <iframe class="iframe-embed iframe-csv" src="assets/marketing_analytics/cleaned_data_sample.html" loading="lazy" title="Cleaned data sample" aria-label="Cleaned data sample"></iframe>
          <div class="downloads"><a class="btn" href="assets/marketing_analytics/cleaned_data_sample.csv" download>Download CSV</a></div>
        </figure>
        <details>
          <summary>Interactive table</summary>
          <iframe class="iframe-embed" style="height:420px" src="assets/marketing_analytics/cleaned_data_table_interactive.html" loading="lazy" title="Interactive cleaned table" aria-label="Interactive cleaned table"></iframe>
        </details>
      `
    },
    // Interactive conversion chart
    {
      type: 'embed',
      title: 'Interactive — Conversion Rates by Group',
      description: 'Explore conversion rates across groups (A/B/C/D) with hover details.',
      tags: ['Plotly', 'Interactivity'],
      contentHtml: `
        <iframe class="iframe-embed" style="height:420px" src="assets/marketing_analytics/conversion_rates_interactive.html" loading="lazy" title="Interactive conversion rates" aria-label="Interactive conversion rates"></iframe>
      `
    },
    // Static visuals (exported from notebook)
    {
      type: 'embed',
      title: 'Static Visuals — Summary',
      description: 'Publication-ready charts exported from the notebook.',
      tags: ['Seaborn', 'Reporting'],
      contentHtml: `
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
  ],
  Healthcare: [
    {
      title: 'Patient Readmission Prediction',
      description: 'Model readmission risk with logistic regression.',
      link: '../Scripts/patient_readmission_model.py',
      tags: ['Classification', 'Healthcare'],
      chartKey: 'Healthcare'
    }
  ],
  Finance: [
    {
      title: 'Stock Market Analysis',
      description: 'Forecast next-day prices for NVDA.',
      link: '../Scripts/stock_market_analysis.py',
      tags: ['Time Series', 'Finance'],
      chartKey: 'Finance'
    }
  ]
};

// Inline SVG icons for each category. Keeping icons minimal avoids external
// dependencies and ensures consistent rendering across browsers.
const categoryIcons = {
  Operations: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z" stroke="currentColor" stroke-width="1.6"/>
      <path d="M4 13h2l1 2 2 1v2l3 2 3-2v-2l2-1 1-2h2l1-2-1-2h-2l-1-2-2-1V4l-3-2-3 2v2l-2 1-1 2H4l-1 2 1 2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
    </svg>`,
  Marketing: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M3 10v4m0-4 12-5v14l-12-5Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M21 9v6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M18 10v4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
    </svg>`,
  Healthcare: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 22c5-4 8-7.5 8-12a8 8 0 1 0-16 0c0 4.5 3 8 8 12Z" stroke="currentColor" stroke-width="1.6"/>
      <path d="M8 12h3l1.2-3.2L14 14h2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`,
  Finance: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 19h16" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M6 15l3-3 3 2 5-6 1 1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`
};

/**
 * Return the SVG markup for a given category.  Defaults to Operations if
 * none is found.
 * @param {string} category
 */
function iconFor(category) {
  return categoryIcons[category] || categoryIcons.Operations;
}

// Chart instances keyed by canvas id to allow clean teardown when re-rendering.
const chartInstances = {};

/*
 * Dashboard statistics are inlined into this script to avoid CORS issues
 * when served over the file:// protocol.  These values are generated
 * offline by running the associated Python scripts and saved as a JSON
 * object.  Embedding them directly also improves performance by
 * eliminating an asynchronous fetch.
 */
const DASHBOARD_DATA = {
  "supply_forecasts": {"A": 204.04752837450314, "B": 190.18559224811838, "C": 223.68580944871164},
  "supply_eoqs": {"A": 90.34324067123188, "B": 87.22054626018307, "C": 94.590868364491},
  "marketing_segment_counts": {"2": 384, "0": 353, "1": 263},
  "marketing_conversion_rates": [0.056451612903225805, 0.06746031746031746],
  "marketing_z_stat": -0.7217637383241345,
  "marketing_p_value": 0.47043974720861326,
  "health_metrics": {"Accuracy": 0.8, "Precision": 0.0, "Recall": 0.0, "ROC_AUC": 0.5550694444444444},
  "finance_dates": [
    "2025-01-25", "2025-01-26", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30",
    "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04", "2025-02-05",
    "2025-02-06", "2025-02-07", "2025-02-08", "2025-02-09", "2025-02-10", "2025-02-11",
    "2025-02-12", "2025-02-13", "2025-02-14", "2025-02-15", "2025-02-16", "2025-02-17",
    "2025-02-18", "2025-02-19", "2025-02-20", "2025-02-21", "2025-02-22", "2025-02-23",
    "2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28", "2025-03-01",
    "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07",
    "2025-03-08", "2025-03-09", "2025-03-10", "2025-03-11", "2025-03-12", "2025-03-13",
    "2025-03-14", "2025-03-15", "2025-03-16", "2025-03-17", "2025-03-18", "2025-03-19",
    "2025-03-20", "2025-03-21", "2025-03-22", "2025-03-23", "2025-03-24", "2025-03-25",
    "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-29", "2025-03-30", "2025-03-31",
    "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-05", "2025-04-06",
    "2025-04-07", "2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11", "2025-04-12",
    "2025-04-13", "2025-04-14", "2025-04-15", "2025-04-16", "2025-04-17", "2025-04-18",
    "2025-04-19", "2025-04-20", "2025-04-21", "2025-04-22", "2025-04-23", "2025-04-24",
    "2025-04-25", "2025-04-26", "2025-04-27", "2025-04-28", "2025-04-29", "2025-04-30",
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05", "2025-05-06",
    "2025-05-07", "2025-05-08", "2025-05-09", "2025-05-10", "2025-05-11", "2025-05-12",
    "2025-05-13", "2025-05-14", "2025-05-15", "2025-05-16", "2025-05-17", "2025-05-18",
    "2025-05-19", "2025-05-20", "2025-05-21", "2025-05-22", "2025-05-23", "2025-05-24",
    "2025-05-25", "2025-05-26", "2025-05-27", "2025-05-28", "2025-05-29", "2025-05-30",
    "2025-05-31", "2025-06-01", "2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05",
    "2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09", "2025-06-10", "2025-06-11",
    "2025-06-12", "2025-06-13", "2025-06-14", "2025-06-15", "2025-06-16", "2025-06-17",
    "2025-06-18", "2025-06-19", "2025-06-20", "2025-06-21", "2025-06-22", "2025-06-23",
    "2025-06-24", "2025-06-25", "2025-06-26", "2025-06-27", "2025-06-28", "2025-06-29",
    "2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-04", "2025-07-05",
    "2025-07-06", "2025-07-07", "2025-07-08", "2025-07-09", "2025-07-10", "2025-07-11",
    "2025-07-12", "2025-07-13", "2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17",
    "2025-07-18", "2025-07-19", "2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23",
    "2025-07-24", "2025-07-25", "2025-07-26", "2025-07-27", "2025-07-28", "2025-07-29",
    "2025-07-30", "2025-07-31", "2025-08-01", "2025-08-02", "2025-08-03", "2025-08-04",
    "2025-08-05", "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-09", "2025-08-10",
    "2025-08-11", "2025-08-12"
  ],
  "finance_close": [
    100.46783308317104, 101.58438880006062, 101.30379463446789, 102.12661015765718,
    103.10867943720024, 100.80596542751627, 101.42548503694243, 102.57647939247036,
    101.34515473606022, 98.8414960701446, 98.88301003431269, 99.74859420818566,
    96.9869357455238, 96.10757278007823, 95.93030739080635, 96.7065174142449,
    97.19832808613931, 98.39653017037044, 98.49528210516708, 99.26715256530905,
    97.09445341779592, 96.97683437109595, 96.07267320267123, 95.15896523549723,
    95.78124666496784, 94.04834248385554, 96.76633203135124, 94.98905922267784,
    94.53851889177139, 94.39132667055588, 94.27195755946377, 93.48960659930383,
    93.87475326461228, 94.63816262618018, 93.36434448222424, 91.48812271711611,
    89.9327620175448, 90.38093287273381, 90.52426710766296, 89.9519261883477,
    89.44182359881741, 88.55072576966198, 88.9648768863699, 87.06645295862307,
    88.36617783365364, 87.91615272305543, 85.92588475261122, 86.22945141652626,
    86.31270375601125, 87.80320841230198, 86.39769708763774, 86.08534486087815,
    85.53435764365281, 85.60741564420904, 85.4639377725317, 85.49126621816303,
    83.77321840533378, 86.39003705638102, 85.35011130108056, 86.13171229347088,
    86.19484528734505, 86.82973993556026, 89.0601444884607, 90.20818540416735,
    89.07264537238814, 89.32481503633491, 89.3394927911204, 91.4493788991527,
    91.59627602175841, 90.17228870447431, 89.10474468383485, 89.03365826987782,
    91.25345604838452, 92.84699761374888, 92.3444414167893, 91.93101943455704,
    93.65511821501957, 94.64397797073305, 95.43811695709176, 93.86482548731,
    93.35362814268376, 94.06318515946039, 94.54234901375344, 94.57368189871215,
    93.77035603983155, 92.59873809826102, 92.29999038400595, 90.51758232721355,
    92.93458289019375, 93.57763792706244, 93.05792025801924, 94.11657192064044,
    93.58488201219356, 93.39222283758124, 93.12232715084217, 92.79642255004481,
    89.8695401259158, 89.06440931729625, 87.08014193869592, 86.81079789960029,
    86.96028820309053, 88.36099670356181, 89.00676212698598, 89.26022668472775,
    88.92458510687315, 89.23933278719446, 88.63184725922778, 88.1927211048812,
    88.37349433169302, 88.43203133048836, 86.57832177532993, 85.69318562927303,
    86.28736977443043, 87.03990776026707, 87.93622549771396, 89.34818982568167,
    89.29956894441946, 89.54384342342235, 89.72288387127782, 91.18075086869631,
    91.08311131951172, 91.06310875734154, 90.27694425212113, 91.68125637490525,
    89.16987297251724, 88.93711101975205, 89.58430479060216, 92.09602218126246,
    91.67370390393346, 92.26662170467613, 90.84339893177248, 91.80995132963236,
    92.51912113114989, 90.70498801021957, 91.5250111011347, 92.30338532714511,
    92.68297452295167, 93.77724456538292, 93.57525630660176, 92.27493172316235,
    90.87617449476402, 89.91687235228278, 90.43872821572062, 89.41986471254108,
    89.65869265493708, 88.81188334334203, 91.15765509984357, 90.50794918951766,
    89.2445386800504, 92.47684641638688, 93.5089709661334, 91.12475470345858,
    89.88249232560669, 88.39436666053314, 90.40649157156267, 91.11075883319263,
    90.02349204857602, 93.0604418443399, 92.60343483560241, 92.99190433638141,
    92.88648294007267, 91.97681506283301, 91.90676396574668, 93.98899675282996,
    92.900035377921, 95.25189104684291, 93.49317050475075, 95.03509044112137,
    95.00672579261833, 94.37955429685442, 92.05199384154334, 92.01001761963205,
    94.56091931334461, 93.75527227494625, 95.37139019704081, 97.39339540084167,
    96.25295281634524, 96.26184735747248, 95.35026444381818, 94.28608657122349,
    93.66409721377592, 92.17814991745911, 93.56201484124352, 91.97492827928193,
    90.99115201585434, 91.60723602777202, 90.99157771577147, 91.0042651721612,
    90.9901294504059, 90.73826915800633, 88.4857181491647, 88.9152717358504,
    88.50041600541061, 89.10810037395393, 89.57755047460327, 89.7307321291958,
    91.51517399080058, 90.54081666780037, 89.48243442960359, 87.49336685044243
  ]
};

// Previously we used a Promise to lazily fetch JSON.  Now that the
// dashboard data is embedded, simply return a resolved promise.
function loadDashboardData() {
  return Promise.resolve(DASHBOARD_DATA);
}

/**
 * Update the active state of the navigation links to reflect the currently
 * selected category.  This sets aria-current appropriately for accessibility.
 * @param {string} category
 */
function setActive(category) {
  document.querySelectorAll('#nav a').forEach((a) => {
    const el = /** @type {HTMLAnchorElement} */ (a);
    const isActive = el.getAttribute('data-category') === category;
    if (isActive) {
      el.setAttribute('aria-current', 'page');
    } else {
      el.removeAttribute('aria-current');
    }
  });
}

/**
 * Render the list of projects for a given category. Cards can be charts
 * or embedded content. After inserting the markup, we render charts
 * for any cards of type 'chart'.
 * @param {string} category
 */
function render(category) {
  const grid = document.getElementById('grid');
  if (!grid) return;
  const list = projects[category] || [];
  setActive(category);
  // Generate card HTML. Chart cards get a unique canvas id.
  const cards = list
    .map((p, idx) => {
      const type = p.type || 'chart';
      const tags = (p.tags || [])
        .map((t) => `<span class="badge">${t}</span>`)
        .join('');
      const canvasId = `chart-${category}-${idx}`;
      const maybeChart = type === 'chart'
        ? `<div class="chart-container"><canvas id="${canvasId}" aria-label="Dashboard chart for ${category}" role="img"></canvas></div>`
        : '';
      const maybeEmbed = type !== 'chart'
        ? `<div class="embed-container">${p.contentHtml || ''}</div>`
        : '';
      const maybeActions = p.link
        ? `<div class="actions"><a class="btn" href="${p.link}" target="_blank" rel="noopener">View Code</a></div>`
        : '';
      return `
        <article class="card">
          <header class="card-header">
            <div class="icon">${iconFor(category)}</div>
            <div>
              <h3>${p.title}</h3>
              <div class="meta">${tags}</div>
            </div>
          </header>
          <p>${p.description || ''}</p>
          ${maybeChart}
          ${maybeEmbed}
          ${maybeActions}
        </article>
      `;
    })
    .join('');
  grid.innerHTML = cards || `<p class="lead">No projects yet in <strong>${category}</strong>.</p>`;
  // Render all charts for this category after the DOM has been updated.
  requestAnimationFrame(() => {
    list.forEach((p, idx) => {
      const type = p.type || 'chart';
      if (type === 'chart') {
        const canvasId = `chart-${category}-${idx}`;
        renderChart(category, canvasId).catch((err) => console.error(err));
      }
    });
  });
}

/**
 * Draw the appropriate chart inside the provided canvas for a given category.
 * If a chart for that canvas already exists it is destroyed before
 * creating a new instance.  The function uses the precomputed JSON
 * summaries to populate chart datasets.
 * @param {string} category
 * @param {string} canvasId
 */
async function renderChart(category, canvasId) {
  const data = await loadDashboardData();
  const canvas = document.getElementById(canvasId);
  if (!(canvas && canvas.getContext)) return;
  // Clean up any existing chart instance for this canvas
  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
  }
  const ctx = canvas.getContext('2d');
  let config;
  if (category === 'Operations') {
    const labels = Object.keys(data.supply_forecasts);
    const forecastVals = labels.map((k) => data.supply_forecasts[k]);
    const eoqVals = labels.map((k) => data.supply_eoqs[k]);
    config = {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Forecast (next month)',
            data: forecastVals,
            backgroundColor: 'rgba(79, 70, 229, 0.6)',
            borderColor: 'rgba(79, 70, 229, 1)',
            borderWidth: 1,
          },
          {
            label: 'EOQ',
            data: eoqVals,
            backgroundColor: 'rgba(129, 140, 248, 0.6)',
            borderColor: 'rgba(129, 140, 248, 1)',
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top' },
          title: {
            display: true,
            text: 'Forecast vs EOQ per Product',
          },
        },
        scales: {
          y: {
            beginAtZero: true,
          },
        },
      },
    };
  } else if (category === 'Marketing') {
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
    config = {
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
  } else if (category === 'Healthcare') {
    const labels = Object.keys(data.health_metrics);
    const values = labels.map((k) => data.health_metrics[k]);
    config = {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Metric Value',
            data: values,
            backgroundColor: 'rgba(79, 70, 229, 0.6)',
            borderColor: 'rgba(79, 70, 229, 1)',
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: 'Readmission Model Performance',
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 1,
          },
        },
      },
    };
  } else if (category === 'Finance') {
    const labels = data.finance_dates;
    const values = data.finance_close;
    config = {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Synthetic NVDA Close',
            data: values,
            fill: false,
            borderColor: 'rgba(79, 70, 229, 1)',
            tension: 0.2,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top' },
          title: {
            display: true,
            text: 'NVDA Closing Price (synthetic)',
          },
        },
        scales: {
          x: {
            ticks: {
              maxTicksLimit: 6,
              callback: function(value, index) {
                const label = this.getLabelForValue(value);
                // Show only every 30th date for readability
                const total = labels.length;
                const step = Math.ceil(total / 6);
                return index % step === 0 ? label : '';
              },
            },
          },
          y: {
            beginAtZero: false,
          },
        },
      },
    };
  } else {
    return;
  }
  chartInstances[canvasId] = new Chart(ctx, config);
}

/**
 * Attach click handlers to the navigation items and handle deep-linking
 * via URL hash.  If the user navigates directly to /#Finance, that
 * category will be selected on page load.  Subsequent hash changes
 * trigger re-renders.
 */
function attachNavHandlers() {
  document.querySelectorAll('#nav a').forEach((anchor) => {
    anchor.addEventListener('click', (e) => {
      e.preventDefault();
      const target = /** @type {HTMLElement} */ (e.currentTarget);
      const category = target.getAttribute('data-category');
      if (category) {
        location.hash = category;
        render(category);
      }
    });
  });
  // load from hash or default
  const initial = (location.hash || '#Operations').replace('#', '');
  render(initial);
  // update on manual hash change
  window.addEventListener('hashchange', () => {
    const cat = (location.hash || '#Operations').replace('#', '');
    render(cat);
  });
}

document.addEventListener('DOMContentLoaded', attachNavHandlers);