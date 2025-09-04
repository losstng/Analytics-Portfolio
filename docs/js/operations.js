/**
 * Operations analytics domain module.
 * Contains templates, chart configuration, and data specific to operations projects.
 */

/**
 * HTML templates for operations project content
 */
export const operationsTemplates = {
  overview: `
    <div class="markdown">
      <p>Advanced Operations Analytics featuring a comprehensive 3D Warehouse Digital Twin with end-to-end supply chain optimization reducing pick distance 22% and cycle time 28%.</p>
      <h4>Technical Innovation Highlights</h4>
      <ul>
        <li><strong>3D Digital Twin</strong>: Interactive warehouse simulation with 1,500 SKUs across 4 levels</li>
        <li><strong>Multi-Layer Optimization</strong>: Forecasting + clustering + MILP + routing integration</li>
        <li><strong>Discrete-Event Simulation</strong>: SimPy validation with 26-week performance tracking</li>
        <li><strong>Spatial Analytics</strong>: 3D slotting optimization with velocity-based zoning</li>
        <li><strong>Process Mining</strong>: Congestion analysis and bottleneck identification</li>
      </ul>
      <h4>Business Impact Methodology</h4>
      <ul>
        <li><strong>ROI Analysis</strong>: 335% return with $191,000 annual savings projection</li>
        <li><strong>Performance Metrics</strong>: Lines/hour (+15%), cycle time (-28%), distance (-22%)</li>
        <li><strong>Causal Validation</strong>: Difference-in-differences analysis for impact isolation</li>
        <li><strong>Implementation Roadmap</strong>: Phased rollout with 3.4-month payback period</li>
      </ul>
    </div>
  `,

  businessImpact: `
    <div class="markdown">
      <h4>Warehouse Optimization Business Impact</h4>
      <div class="metrics-grid">
        <div class="metric-card">
          <span class="metric-value">-22%</span>
          <span class="metric-label">Pick Distance Reduction</span>
          <span class="metric-change">245m → 191m average</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">+15%</span>
          <span class="metric-label">Productivity Increase</span>
          <span class="metric-change">47 → 54 lines/hour</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">-28%</span>
          <span class="metric-label">Cycle Time Reduction</span>
          <span class="metric-change">18.3 → 13.2 minutes P95</span>
        </div>
        <div class="metric-card">
          <span class="metric-value">335%</span>
          <span class="metric-label">ROI</span>
          <span class="metric-change">$191K annual savings</span>
        </div>
      </div>
    </div>
  `,

  warehouse3D: `
    <figure>
      <figcaption>3D Warehouse Digital Twin - Interactive Optimization Visualization</figcaption>
      <iframe class="iframe-embed" style="height:700px" src="assets/operations_analytics/warehouse_3d_interactive.html?v=${Date.now()}" loading="lazy" title="3D Warehouse Digital Twin" aria-label="Interactive 3D warehouse showing before and after optimization with SKU velocity color-coding"></iframe>
    </figure>
  `,

  velocityPareto: `
    <figure>
      <figcaption>SKU Velocity Distribution - ABC Pareto Analysis</figcaption>
      <iframe class="iframe-embed" style="height:600px" src="assets/operations_analytics/velocity_pareto_interactive.html?v=${Date.now()}" loading="lazy" title="SKU Velocity Pareto" aria-label="Pareto chart showing SKU velocity distribution and ABC classification patterns"></iframe>
    </figure>
  `,

  pickPathComparison: `
    <figure>
      <figcaption>Pick Path Optimization Analysis - Before vs After</figcaption>
      <iframe class="iframe-embed" style="height:800px" src="assets/operations_analytics/pick_path_comparison_interactive.html?v=${Date.now()}" loading="lazy" title="Pick Path Comparison" aria-label="Comprehensive analysis of pick path distance improvements showing distribution, scatter plots, and cumulative performance"></iframe>
    </figure>
  `,

  congestionHeatmap: `
    <figure>
      <figcaption>Warehouse Congestion Analysis - Peak Hour Patterns</figcaption>
      <iframe class="iframe-embed" style="height:600px" src="assets/operations_analytics/congestion_heatmap_interactive.html?v=${Date.now()}" loading="lazy" title="Congestion Heatmap" aria-label="Heatmap showing warehouse congestion patterns by aisle and hour with peak period identification"></iframe>
    </figure>
  `,

  simulationDashboard: `
    <figure>
      <figcaption>Performance Dashboard - 26 Week Simulation Results</figcaption>
      <iframe class="iframe-embed" style="height:800px" src="assets/operations_analytics/simulation_dashboard_interactive.html?v=${Date.now()}" loading="lazy" title="Simulation Dashboard" aria-label="Comprehensive performance dashboard showing KPI trends over 26-week simulation period"></iframe>
    </figure>
  `,

  costWaterfall: `
    <figure>
      <figcaption>Business Impact Analysis - Annual Cost Savings Waterfall</figcaption>
      <iframe class="iframe-embed" style="height:600px" src="assets/operations_analytics/cost_waterfall_interactive.html?v=${Date.now()}" loading="lazy" title="Cost Waterfall" aria-label="Waterfall chart showing breakdown of annual cost savings and implementation costs with ROI calculation"></iframe>
    </figure>
  `,

  executiveSummary: `
    <figure>
      <figcaption>Executive Summary - Key Performance Metrics Comparison</figcaption>
      <iframe class="iframe-embed" style="height:500px" src="assets/operations_analytics/executive_summary_table.html?v=${Date.now()}" loading="lazy" title="Executive Summary" aria-label="Executive summary table comparing baseline vs optimized performance metrics"></iframe>
    </figure>
  `,

  technicalMethodology: `
    <div class="methodology-section">
      <h4>Technical Implementation Stack</h4>
      <div class="tech-grid">
        <div class="tech-item">
          <h5>Data Generation</h5>
          <p>Synthetic 3D warehouse (1,500 SKUs, 10K locations) with realistic velocity distributions, order patterns, and spatial constraints using Python Faker + controlled distributions.</p>
        </div>
        <div class="tech-item">
          <h5>Optimization Engine</h5>
          <p>Multi-stage MILP optimization: SKU clustering by affinity → velocity-based zone assignment → routing optimization with OR-Tools for TSP variants.</p>
        </div>
        <div class="tech-item">
          <h5>Simulation Validation</h5>
          <p>Discrete-event simulation (SimPy) with stochastic pick times, congestion modeling, and 26-week performance tracking for statistical validation.</p>
        </div>
        <div class="tech-item">
          <h5>Visualization Pipeline</h5>
          <p>Interactive 3D visualization (Plotly), heatmaps, Pareto analysis, and executive dashboards with embedded data for seamless web integration.</p>
        </div>
      </div>
    </div>
  `,

  keyFindings: `
    <div class="findings-section">
      <h4>Key Analytical Findings</h4>
      <div class="findings-grid">
        <div class="finding-item">
          <h5>Velocity Concentration</h5>
          <p>20% of SKUs generate 80% of picking activity (classic Pareto), but baseline slotting ignored this pattern, placing high-velocity items in distant bulk zones.</p>
        </div>
        <div class="finding-item">
          <h5>Congestion Hotspots</h5>
          <p>Peak hours (9-11 AM, 1-3 PM) create 2.3x congestion in fast zones. Optimized layout redistributes load and reduces bottlenecks by 20%.</p>
        </div>
        <div class="finding-item">
          <h5>Distance-Velocity Correlation</h5>
          <p>Strong negative correlation (-0.67) between SKU velocity and distance to dock in optimized layout vs random pattern (0.03) in baseline.</p>
        </div>
        <div class="finding-item">
          <h5>Simulation Validation</h5>
          <p>Monte Carlo simulation (500 runs) confirms 22% distance reduction with 95% confidence interval [19.8%, 24.2%]. Results statistically significant (p<0.001).</p>
        </div>
      </div>
    </div>
  `
};

/**
 * Operations project configuration
 */
export const operationsProjects = [
  {
    id: 'overview',
    title: 'Operations Analytics Overview',
    description: 'Advanced 3D Warehouse Digital Twin with comprehensive supply chain optimization',
    tags: ['Overview', 'Digital Twin', 'Business Impact'],
    type: 'embed',
    category: 'operations',
    content: 'overview',
    contentHtml: operationsTemplates.overview
  },
  {
    id: 'business-impact',
    title: 'Business Impact Metrics',
    description: 'ROI analysis showing 335% return with $191K annual savings and key performance improvements',
    tags: ['ROI', 'Business Metrics', 'Performance'],
    type: 'embed',
    category: 'operations',
    content: 'businessImpact',
    contentHtml: operationsTemplates.businessImpact
  },
  {
    id: 'warehouse-digital-twin',
    title: '3D Warehouse Digital Twin',
    description: 'Interactive 3D visualization showing before/after optimization with SKU velocity analysis',
    tags: ['3D Modeling', 'Digital Twin', 'Optimization', 'Interactive'],
    type: 'embed',
    category: 'operations',
    content: 'warehouse3D',
    contentHtml: operationsTemplates.warehouse3D
  },
  {
    id: 'velocity-analysis',
    title: 'SKU Velocity Distribution',
    description: 'ABC Pareto analysis revealing critical velocity patterns and optimization opportunities',
    tags: ['Pareto Analysis', 'ABC Classification', 'Data Visualization'],
    type: 'embed', 
    category: 'operations',
    content: 'velocityPareto',
    contentHtml: operationsTemplates.velocityPareto
  },
  {
    id: 'pick-path-optimization',
    title: 'Pick Path Distance Analysis',
    description: 'Comprehensive before/after analysis of picking route efficiency improvements',
    tags: ['Route Optimization', 'Performance Analysis', 'Statistical Validation'],
    type: 'embed',
    category: 'operations', 
    content: 'pickPathComparison',
    contentHtml: operationsTemplates.pickPathComparison
  },
  {
    id: 'congestion-analysis',
    title: 'Warehouse Congestion Patterns',
    description: 'Aisle-level congestion analysis and peak hour bottleneck identification',
    tags: ['Congestion Modeling', 'Heatmap Analysis', 'Process Mining'],
    type: 'embed',
    category: 'operations',
    content: 'congestionHeatmap',
    contentHtml: operationsTemplates.congestionHeatmap
  },
  {
    id: 'performance-dashboard',
    title: '26-Week Simulation Results',
    description: 'Comprehensive KPI tracking and performance validation over extended simulation period',
    tags: ['Simulation', 'KPI Dashboard', 'Time Series Analysis'],
    type: 'embed',
    category: 'operations',
    content: 'simulationDashboard',
    contentHtml: operationsTemplates.simulationDashboard
  },
  {
    id: 'cost-benefit-analysis',
    title: 'ROI Waterfall Analysis',
    description: 'Detailed breakdown of $191K annual savings with 335% return calculation',
    tags: ['ROI Analysis', 'Cost-Benefit', 'Business Case'],
    type: 'embed',
    category: 'operations',
    content: 'costWaterfall',
    contentHtml: operationsTemplates.costWaterfall
  },
  {
    id: 'executive-summary',
    title: 'Executive Performance Summary',
    description: 'Key metrics comparison table for executive decision-making',
    tags: ['Executive Summary', 'KPI Tracking', 'Performance Metrics'],
    type: 'embed',
    category: 'operations',
    content: 'executiveSummary',
    contentHtml: operationsTemplates.executiveSummary
  },
  {
    id: 'technical-methodology',
    title: 'Technical Implementation',
    description: 'Detailed methodology covering optimization algorithms, simulation, and validation',
    tags: ['Methodology', 'Technical Deep-dive', 'Implementation'],
    type: 'embed',
    category: 'operations',
    content: 'technicalMethodology',
    contentHtml: operationsTemplates.technicalMethodology
  },
  {
    id: 'analytical-findings',
    title: 'Key Analytical Insights',
    description: 'Critical findings from velocity analysis, congestion patterns, and performance validation',
    tags: ['Insights', 'Analytics', 'Key Findings'],
    type: 'embed', 
    category: 'operations',
    content: 'keyFindings',
    contentHtml: operationsTemplates.keyFindings
  },
  {
    id: 'operations-chart',
    title: 'Performance Metrics Chart',
    description: 'Interactive chart comparing baseline vs optimized warehouse performance',
    tags: ['Chart.js', 'Performance Comparison', 'Interactive'],
    type: 'chart',
    category: 'operations',
    content: 'chart'
  }
];

/**
 * Chart creation function for operations analytics.
 * Returns Chart.js configuration based on data type.
 * @param {Object} data - The complete dashboard data object
 * @returns {Object} Chart.js configuration object
 */
export function createOperationsChart(data) {
  // Performance metrics comparison chart
  const performanceMetrics = data.operations_performance_metrics;
  
  return {
    type: 'bar',
    data: {
      labels: ['Pick Distance (m)', 'Lines/Hour', 'Cycle Time (min)', 'Labor Cost ($K)'],
      datasets: [
        {
          label: 'Baseline',
          data: [
            performanceMetrics.pick_distance.baseline,
            performanceMetrics.lines_per_hour.baseline,
            performanceMetrics.cycle_time.baseline,
            performanceMetrics.labor_cost.baseline / 1000
          ],
          backgroundColor: '#ff6b6b',
          borderColor: '#ee5a5a',
          borderWidth: 1
        },
        {
          label: 'Optimized',
          data: [
            performanceMetrics.pick_distance.optimized,
            performanceMetrics.lines_per_hour.optimized,
            performanceMetrics.cycle_time.optimized,
            performanceMetrics.labor_cost.optimized / 1000
          ],
          backgroundColor: '#4ecdc4',
          borderColor: '#45b7b8',
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Warehouse Optimization - Performance Metrics Comparison'
        },
        legend: {
          position: 'top'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Value'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Metrics'
          }
        }
      }
    }
  };
}
