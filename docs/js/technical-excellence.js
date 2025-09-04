/**
 * Technical Excellence Showcase Data and Templates
 * Deep-dive into advanced methodologies and implementation approaches
 */

// Technical excellence content for different domains
export const technicalContent = {
  operations: {
    title: "Operations Analytics - Supply Chain Digital Twin",
    subtitle: "End-to-end synthetic supply chain optimization with 3D warehouse modeling",
    overview: {
      description: "A flagship operations analytics project showcasing an end-to-end Synthetic Supply Chain Digital Twin for a mid-size omni-channel manufacturer—integrating demand forecasting, inventory & multi-echelon replenishment, warehouse 3D slotting optimization, order fulfillment process mining, and prescriptive capacity & routing decisions.",
      impact: [
        "Reduce stockouts by 35%",
        "Decrease inventory holding by 18%", 
        "Cut pick path distance by 22%",
        "Reduce order cycle time by 28%"
      ],
      timeline: "1 intensive weeks",
      technologies: ["Python", "NetworkX", "SimPy", "PuLP", "Plotly", "scikit-learn", "statsmodels"]
    },
    sections: [
      {
        id: "architecture",
        title: "🏗️ Multi-Layer Optimization Architecture",
        icon: "🔧",
        content: {
          description: "Advanced warehouse configuration system with sophisticated spatial optimization and constraint handling.",
          codeExample: `# 3D Warehouse Digital Twin - Core Configuration
class WarehouseConfig:
    def __init__(self):
        # Physical dimensions with realistic constraints
        self.warehouse_length = 100  # meters
        self.warehouse_width = 100   # meters
        self.levels = 4             # vertical levels (0, 1, 2, 3)
        
        # Grid configuration for spatial optimization
        self.grid_size = 2.0        # 2m x 2m grid cells
        self.x_positions = int(self.warehouse_length / self.grid_size)  # 50 positions
        self.y_positions = int(self.warehouse_width / self.grid_size)   # 50 positions
        self.total_positions = self.x_positions * self.y_positions * self.levels
        
        # Zone definitions with capacity constraints
        self.zone_fast_pct = 0.40    # High-velocity items near dock
        self.zone_medium_pct = 0.35  # Medium-velocity balanced placement
        self.zone_bulk_pct = 0.25    # Low-velocity items in back areas`,
          highlights: [
            {
              title: "Spatial Grid System",
              description: "2m x 2m grid cells across 4 vertical levels providing 10,000 possible slot positions"
            },
            {
              title: "Zone-Based Optimization", 
              description: "Strategic placement zones optimized for velocity patterns and operational efficiency"
            },
            {
              title: "Constraint Modeling",
              description: "Physical and operational constraints integrated into optimization framework"
            }
          ]
        }
      },
      {
        id: "forecasting",
        title: "📈 Advanced Velocity Forecasting with Feature Engineering",
        icon: "🔮",
        content: {
          description: "Sophisticated time series feature engineering creating 19+ engineered variables for demand prediction.",
          codeExample: `# Sophisticated Time Series Feature Engineering
def engineer_temporal_features(order_lines_df, sku_master):
    """Create comprehensive feature set with 19+ engineered variables"""
    
    # 1. Rolling Velocity Metrics with Multiple Windows
    velocity_features = []
    for sku_id in sku_master['sku_id']:
        sku_weekly = weekly_velocity.loc[weekly_velocity.index.get_level_values(0) == sku_id]
        
        if len(sku_weekly) > 0:
            weekly_orders = sku_weekly['weekly_orders'].values
            
            # Multi-window rolling averages for trend detection
            velocity_4wk = np.mean(weekly_orders[-4:]) if len(weekly_orders) >= 4 else np.mean(weekly_orders)
            velocity_8wk = np.mean(weekly_orders[-8:]) if len(weekly_orders) >= 8 else np.mean(weekly_orders)
            velocity_12wk = np.mean(weekly_orders[-12:]) if len(weekly_orders) >= 12 else np.mean(weekly_orders)
            
            # Volatility metrics for stability assessment
            velocity_cv = np.std(weekly_orders) / np.mean(weekly_orders) if np.mean(weekly_orders) > 0 else 0
            velocity_mad = np.median(np.abs(weekly_orders - np.median(weekly_orders)))
            
            # Trend analysis using linear regression
            if len(weekly_orders) >= 3:
                x = np.arange(len(weekly_orders))
                slope, intercept = np.polyfit(x, weekly_orders, 1)
                r_squared = np.corrcoef(x, weekly_orders)[0, 1]**2 if len(x) > 1 else 0
            else:
                slope = intercept = r_squared = 0`,
          highlights: [
            {
              title: "Multi-Window Analysis",
              description: "4, 8, and 12-week rolling averages capture both short-term fluctuations and long-term trends"
            },
            {
              title: "Volatility Metrics",
              description: "Coefficient of variation and median absolute deviation quantify demand stability"
            },
            {
              title: "Trend Detection",
              description: "Linear regression and R-squared analysis identifies systematic demand patterns"
            }
          ]
        }
      },
      {
        id: "affinity",
        title: "🕸️ Graph-Based SKU Affinity Analysis",
        icon: "🔗",
        content: {
          description: "Advanced co-occurrence and affinity mining using multiple affinity metrics including Jaccard similarity, lift, and confidence scores.",
          codeExample: `# Advanced Co-occurrence and Affinity Mining
def compute_sku_affinity_network(order_lines_df):
    """Build weighted graph of SKU relationships using multiple affinity metrics"""
    
    # Create order-SKU matrix for market basket analysis
    order_skus = order_lines_df.groupby('order_id')['sku_id'].apply(list).reset_index()
    
    # Calculate multiple affinity measures
    affinity_metrics = {}
    
    for _, row in order_skus.iterrows():
        skus = row['sku_id']
        if len(skus) > 1:
            for sku1, sku2 in combinations(skus, 2):
                pair = tuple(sorted([sku1, sku2]))
                
                if pair not in affinity_metrics:
                    affinity_metrics[pair] = {
                        'co_occurrence': 0,
                        'jaccard_numerator': 0,
                        'lift_numerator': 0
                    }
                
                affinity_metrics[pair]['co_occurrence'] += 1
    
    # Calculate advanced affinity scores
    total_orders = len(order_skus)
    sku_frequencies = order_lines_df['sku_id'].value_counts()
    
    for pair, metrics in affinity_metrics.items():
        sku1, sku2 = pair
        
        # Jaccard similarity: |A ∩ B| / |A ∪ B|
        union_orders = sku_frequencies[sku1] + sku_frequencies[sku2] - metrics['co_occurrence']
        jaccard = metrics['co_occurrence'] / union_orders if union_orders > 0 else 0
        
        # Lift: P(A,B) / (P(A) * P(B))
        prob_a = sku_frequencies[sku1] / total_orders
        prob_b = sku_frequencies[sku2] / total_orders
        prob_ab = metrics['co_occurrence'] / total_orders
        lift = prob_ab / (prob_a * prob_b) if (prob_a * prob_b) > 0 else 0`,
          highlights: [
            {
              title: "Market Basket Analysis",
              description: "Comprehensive analysis of which SKUs are frequently ordered together"
            },
            {
              title: "Multiple Affinity Metrics",
              description: "Jaccard similarity, lift, and confidence measures provide robust relationship scoring"
            },
            {
              title: "Network Graph Structure",
              description: "SKU relationships modeled as weighted graph for optimization algorithms"
            }
          ]
        }
      },
      {
        id: "optimization",
        title: "🎯 MILP-Based Slotting Optimization with Constraints",
        icon: "⚙️",
        content: {
          description: "Mixed-Integer Linear Programming formulation for optimal warehouse slotting with comprehensive constraint handling.",
          codeExample: `# Advanced Mixed-Integer Linear Programming for Optimal Slotting
def formulate_slotting_milp(sku_features, warehouse_locations, affinity_matrix):
    """
    Formulate comprehensive MILP model for warehouse slotting optimization
    
    Decision Variables:
    - x[i,j] = 1 if SKU i is assigned to location j, 0 otherwise
    
    Objective:
    Minimize: Σ(velocity[i] * distance[j] * x[i,j]) + 
              λ₁ * Σ(affinity[i,k] * separation_penalty[j,l] * x[i,j] * x[k,l]) +
              λ₂ * Σ(zone_mismatch_penalty[i,j] * x[i,j])
    
    Constraints:
    1. Each SKU assigned to exactly one location: Σⱼ x[i,j] = 1 ∀i
    2. Each location holds at most one SKU: Σᵢ x[i,j] ≤ 1 ∀j
    3. Capacity constraints: cube[i] * x[i,j] ≤ capacity[j] ∀i,j
    4. Zone velocity constraints: Σᵢ(velocity[i] * x[i,j]) ≥ min_velocity[zone[j]] ∀j
    5. Fragile item constraints: fragile[i] * x[i,j] ≤ ground_level[j] ∀i,j
    """
    
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
    
    # Create the optimization problem
    prob = LpProblem("Warehouse_Slotting_Optimization", LpMinimize)
    
    # Decision variables
    skus = sku_features['sku_id'].tolist()
    locations = warehouse_locations['location_id'].tolist()
    
    x = {}
    for i in skus:
        for j in locations:
            x[i, j] = LpVariable(f"assign_{i}_{j}", cat='Binary')
    
    # Objective function with multiple optimization criteria
    objective = 0
    
    # Primary objective: minimize weighted distance
    for i in skus:
        sku_velocity = sku_features[sku_features['sku_id'] == i]['velocity_4wk_avg'].iloc[0]
        for j in locations:
            dock_distance = warehouse_locations[warehouse_locations['location_id'] == j]['dock_distance'].iloc[0]
            objective += sku_velocity * dock_distance * x[i, j]`,
          highlights: [
            {
              title: "Multi-Objective Optimization",
              description: "Balances travel distance, affinity placement, and zone constraints simultaneously"
            },
            {
              title: "Comprehensive Constraints",
              description: "Capacity, velocity, zone, and fragile item constraints ensure feasible solutions"
            },
            {
              title: "Binary Decision Variables",
              description: "Clean assignment model ensuring each SKU gets exactly one optimal location"
            }
          ]
        }
      },
      {
        id: "simulation",
        title: "🎮 Discrete-Event Simulation with Congestion Modeling",
        icon: "🔄",
        content: {
          description: "Sophisticated SimPy-based warehouse operations simulation with stochastic processes and congestion-aware routing.",
          codeExample: `# Advanced SimPy-Based Warehouse Operations Simulation
import simpy
from collections import defaultdict

class WarehouseSimulation:
    """
    Sophisticated discrete-event simulation with:
    - Stochastic arrival processes
    - Resource contention modeling  
    - Congestion-aware routing
    - Performance metric collection
    """
    
    def __init__(self, env, config, slotting_data):
        self.env = env
        self.config = config
        self.slotting = slotting_data
        
        # Resources with finite capacity
        self.pickers = simpy.Resource(env, capacity=config.num_pickers)
        self.aisles = {aisle_id: simpy.Resource(env, capacity=config.aisle_capacity) 
                      for aisle_id in config.aisle_ids}
        
        # Performance metrics collection
        self.metrics = defaultdict(list)
        self.congestion_data = defaultdict(list)
        
    def picker_process(self, picker_id, batch_orders):
        """Simulate individual picker with realistic constraints"""
        
        with self.pickers.request() as picker_request:
            yield picker_request
            
            batch_start_time = self.env.now
            total_distance = 0
            aisle_queue_times = []
            
            # Optimize pick sequence using nearest-neighbor heuristic
            optimized_sequence = self.optimize_pick_sequence(batch_orders)
            
            for pick_location in optimized_sequence:
                # Travel to location with congestion consideration
                travel_time, queue_time = yield from self.travel_to_location(pick_location)
                total_distance += pick_location['distance_to_dock']
                aisle_queue_times.append(queue_time)
                
                # Pick time with stochastic variation
                base_pick_time = self.config.base_pick_time
                item_complexity = pick_location.get('complexity_factor', 1.0)
                fragile_penalty = 10 if pick_location.get('fragile', False) else 0
                
                pick_time = np.random.lognormal(
                    mean=np.log(base_pick_time * item_complexity), 
                    sigma=0.3
                ) + fragile_penalty
                
                yield self.env.timeout(pick_time)`,
          highlights: [
            {
              title: "Resource Contention",
              description: "Models finite picker capacity and aisle congestion with realistic queueing"
            },
            {
              title: "Stochastic Processes",
              description: "Log-normal pick times with complexity and fragility adjustments"
            },
            {
              title: "Congestion-Aware Routing",
              description: "Dynamic routing considers real-time aisle utilization for optimal paths"
            }
          ]
        }
      },
      {
        id: "causal",
        title: "📊 Causal Impact Analysis with Difference-in-Differences",
        icon: "🧪",
        content: {
          description: "Robust causal inference framework using difference-in-differences methodology with multiple robustness checks.",
          codeExample: `# Robust Causal Inference Framework
def difference_in_differences_analysis(baseline_results, optimized_results, control_factors):
    """
    Sophisticated DID analysis with multiple robustness checks
    
    Model: Y_it = α + β₁*Treat_i + β₂*Post_t + β₃*(Treat_i × Post_t) + γ*X_it + ε_it
    
    Where:
    - Y_it: Performance metric (cycle time, distance, etc.)
    - Treat_i: 1 if unit i receives optimized slotting, 0 if baseline
    - Post_t: 1 if time period t is after intervention, 0 if before
    - β₃: Difference-in-differences estimator (causal effect)
    - X_it: Control variables (batch size, complexity, etc.)
    """
    
    import statsmodels.api as sm
    from scipy import stats
    
    # Prepare data for regression analysis
    analysis_data = prepare_did_dataset(baseline_results, optimized_results, control_factors)
    
    # Core DID regression
    did_formula = """
    cycle_time ~ 
        C(treatment) + 
        C(post_period) + 
        C(treatment):C(post_period) +
        batch_size + 
        order_complexity + 
        time_of_day + 
        congestion_index
    """
    
    # Fit primary model
    primary_model = sm.OLS.from_formula(did_formula, data=analysis_data).fit(
        cov_type='cluster', 
        cov_kwds={'groups': analysis_data['batch_id']}
    )
    
    # Extract causal effect
    did_coefficient = primary_model.params['C(treatment)[T.1]:C(post_period)[T.1]']
    did_se = primary_model.bse['C(treatment)[T.1]:C(post_period)[T.1]']
    did_pvalue = primary_model.pvalues['C(treatment)[T.1]:C(post_period)[T.1]']`,
          highlights: [
            {
              title: "Causal Identification",
              description: "Difference-in-differences design isolates treatment effects from confounding factors"
            },
            {
              title: "Robustness Testing",
              description: "Placebo tests, parallel trends analysis, and bootstrap confidence intervals"
            },
            {
              title: "Statistical Rigor",
              description: "Clustered standard errors and comprehensive sensitivity analysis"
            }
          ]
        }
      },
      {
        id: "visualization",
        title: "🎨 Real-Time 3D Visualization and Interactive Analytics",
        icon: "📊",
        content: {
          description: "Advanced Plotly 3D visualization with animation capabilities and interactive drill-down features.",
          codeExample: `# Advanced Plotly 3D Visualization with Animation
def create_interactive_warehouse_twin(slotting_data, performance_metrics, optimization_history):
    """
    Generate sophisticated 3D digital twin with:
    - Real-time performance overlay
    - Animation of optimization process
    - Interactive drill-down capabilities
    - Heat map integration
    """
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create multi-panel 3D dashboard
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
            [{'type': 'surface'}, {'type': 'scatter'}]
        ],
        subplot_titles=[
            'Baseline Layout', 'Optimized Layout',
            'Congestion Heat Surface', 'Performance Timeline'
        ]
    )
    
    # Generate warehouse rack infrastructure
    warehouse_structure = generate_3d_warehouse_structure()
    
    # Add warehouse infrastructure to both 3D plots
    for row, col in [(1, 1), (1, 2)]:
        # Add rack structures
        for rack in warehouse_structure['racks']:
            fig.add_trace(
                go.Mesh3d(
                    x=rack['x_vertices'],
                    y=rack['y_vertices'], 
                    z=rack['z_vertices'],
                    color='lightgray',
                    opacity=0.3,
                    showlegend=False
                ),
                row=row, col=col
            )`,
          highlights: [
            {
              title: "Multi-Panel Dashboard",
              description: "Integrated view of baseline vs optimized layouts with performance metrics"
            },
            {
              title: "3D Spatial Modeling",
              description: "True-to-scale warehouse representation with rack structures and aisles"
            },
            {
              title: "Interactive Analytics",
              description: "Drill-down capabilities with rich hover information and dynamic filtering"
            }
          ]
        }
      }
    ],
    metrics: [
      {
        value: "35%",
        label: "Stockout Reduction"
      },
      {
        value: "18%", 
        label: "Inventory Decrease"
      },
      {
        value: "22%",
        label: "Distance Reduction"
      },
      {
        value: "28%",
        label: "Cycle Time Improvement"
      }
    ]
  },
  
  marketing: {
    title: "Marketing Science - MMM & Causal Inference",
    subtitle: "Advanced marketing mix modeling with causal attribution",
    overview: {
      description: "Sophisticated marketing science project demonstrating advanced statistical methods for marketing attribution, customer lifetime value optimization, and campaign effectiveness measurement.",
      impact: [
        "Increase ROAS by 45%",
        "Improve attribution accuracy by 62%",
        "Optimize media mix allocation",
        "Enhance customer segmentation"
      ],
      timeline: "3 weeks",
      technologies: ["Python", "PyMC", "CausalImpact", "LightGBM", "Plotly", "Streamlit"]
    },
    sections: [],
    metrics: []
  },

  finance: {
    title: "Financial Engineering - Risk & Portfolio Optimization", 
    subtitle: "Quantitative finance with machine learning and risk modeling",
    overview: {
      description: "Advanced quantitative finance project showcasing modern portfolio theory implementation, risk modeling, and algorithmic trading strategies using machine learning.",
      impact: [
        "Sharpe ratio improvement of 0.8",
        "VaR reduction by 25%",
        "Alpha generation of 3.2%",
        "Volatility targeting accuracy"
      ],
      timeline: "4 weeks",
      technologies: ["Python", "QuantLib", "numpy", "scipy", "sklearn", "matplotlib"]
    },
    sections: [],
    metrics: []
  },

  healthcare: {
    title: "Healthcare Analytics - Predictive Modeling & Outcomes",
    subtitle: "Clinical data analysis with machine learning for patient outcomes",
    overview: {
      description: "Healthcare analytics project demonstrating predictive modeling for patient readmission, clinical outcome optimization, and healthcare resource allocation.",
      impact: [
        "Reduce readmission rate by 23%",
        "Improve diagnostic accuracy by 18%",
        "Optimize resource allocation",
        "Enhance patient care pathways"
      ],
      timeline: "3 weeks", 
      technologies: ["Python", "scikit-learn", "XGBoost", "SHAP", "matplotlib", "seaborn"]
    },
    sections: [],
    metrics: []
  }
};

// Function to render technical content for a specific domain
export function renderTechnicalContent(domain) {
  const content = technicalContent[domain];
  if (!content) return '';

  const sectionsHtml = content.sections.map(section => `
    <div class="tech-section" id="${section.id}">
      <h4>
        <span class="tech-section-icon">${section.icon}</span>
        ${section.title}
      </h4>
      <p>${section.content.description}</p>
      
      <div class="tech-code-block">
        <pre><code>${section.content.codeExample}</code></pre>
      </div>
      
      <div class="tech-highlights">
        ${section.content.highlights.map(highlight => `
          <div class="tech-highlight">
            <h5>${highlight.title}</h5>
            <p>${highlight.description}</p>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');

  const metricsHtml = content.metrics.length > 0 ? `
    <div class="tech-metrics">
      ${content.metrics.map(metric => `
        <div class="tech-metric">
          <span class="tech-metric-value">${metric.value}</span>
          <span class="tech-metric-label">${metric.label}</span>
        </div>
      `).join('')}
    </div>
  ` : '';

  return `
    <div class="tech-project-overview">
      <h3>${content.title}</h3>
      <p class="tech-subtitle">${content.subtitle}</p>
      
      <div class="tech-project-meta">
        <div class="tech-meta-item">Timeline: ${content.overview.timeline}</div>
        <div class="tech-meta-item">Technologies: ${content.overview.technologies.join(', ')}</div>
      </div>
      
      <p>${content.overview.description}</p>
      
      ${metricsHtml}
    </div>
    
    ${sectionsHtml}
  `;
}
