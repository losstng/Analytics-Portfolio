# Analytics Portfolio - AI Agent Instructions

## Project Overview
This is a **data analytics portfolio** showcasing domain expertise across **Marketing, Finance, Healthcare, and Operations** through synthetic datasets, ML modeling, and interactive web visualizations. The project combines Python data science workflows with a TypeScript/ES6 modular portfolio website.

## Architecture & Key Components

### **Data Science Layer** (`/docs/Scripts/`, `/Notebooks/`, `/python.py`)
- **Pattern**: Each domain script follows `generate_data() → extract() → transform() → load()` + modeling
- **Domain Scripts**: `marketing_analytics.py`, `supply_chain_optimization.py`, `patient_readmission_model.py`, `stock_market_analysis.py`
- **Data Flow**: Raw CSV → cleaning/feature engineering → ML models → results exported for web visualization
- **Key Dependencies**: pandas, sklearn, statsmodels, pulp (optimization), xgboost

### **Portfolio Website** (`/docs/`)
- **Architecture**: Domain-driven modular ES6 structure with Chart.js visualizations
- **Entry Point**: `index.html` → `main.js` → domain modules (`marketing.js`, `finance.js`, etc.)
- **Data Embedding**: `js/data.js` contains `DASHBOARD_DATA` - all chart data is embedded (no backend required)
- **Navigation**: CV landing page + dropdown to domain portfolios with project cards

### **Domain Module Pattern** (`docs/js/{domain}.js`)
Each domain exports:
```javascript
export const {domain}Projects = [/* project metadata */];
export function create{Domain}Chart(data) { /* Chart.js config */ }
```

## Development Workflows

### **Running the Portfolio Site**
```powershell
cd docs
python -m http.server 8000  # Required - file:// breaks ES modules
# Navigate to http://127.0.0.1:8000/
```

### **Python Environment**
- **Virtual Environment**: `.venv/` (Python 3.13)
- **Activation**: `.venv\Scripts\activate` (Windows)
- **Package Management**: Use `pip install` in activated environment

### **Data Science Workflow**
1. **Generate synthetic data**: Run scripts in `/docs/Scripts/` 
2. **Exploratory analysis**: Use `/Notebooks/Master.ipynb` or `/Notebooks/Showcase.ipynb`
3. **Export results**: Scripts auto-save cleaned data to `/generated_data/`
4. **Web integration**: Update `js/data.js` with key metrics for dashboard

## Project-Specific Conventions

### **File Organization**
- **Raw data**: `/raw_data/` (stock CSVs, intraday data)
- **Generated data**: `/generated_data/` (cleaned outputs from scripts)
- **Static assets**: `/docs/assets/` (images, exported HTML tables)
- **SQL practice**: `/SQL` (single file with Classic Models DB queries)

### **Code Style**
- **Python**: Type hints, docstrings, `pathlib.Path` for file handling
- **JavaScript**: ES6 modules, destructuring, async/await for chart rendering
- **Data Processing**: Consistent `RNG = np.random.default_rng(seed=42)` for reproducibility

### **Chart Integration Pattern**
```javascript
// In domain modules: return Chart.js config objects
export function createMarketingChart(data) {
  return {
    type: 'bar',
    data: data.marketing.conversion_rates,
    options: { /* responsive config */ }
  };
}
```

### **Project Card Types**
- `type: 'chart'` → Canvas with Chart.js visualization
- `type: 'embed'` → HTML content (tables, iframes)
- Always include `title`, `description`, `tags[]`, optional `link`

## Integration Points

### **Data Dependencies**
- **Stock data**: yfinance → `/raw_data/intraday/` (via `intradaydata.py`)
- **ML models**: sklearn pipelines → pickle serialization
- **Web charts**: Python analysis → JSON export → `js/data.js` embedding

### **Cross-Component Communication**
- **Navigation**: `main.js` manages view state (`currentView`, `currentCategory`)
- **Chart rendering**: `charts.js` delegates to domain-specific chart creators
- **Module loading**: `templates.js` aggregates all domain projects

## Common Debugging

### **Website Issues**
- **Navigation not working**: Ensure HTTP server (not file://)
- **Charts not rendering**: Check browser console for ES module errors
- **Missing data**: Verify `js/data.js` has required data structure

### **Python Issues**
- **Import errors**: Activate virtual environment first
- **Data generation**: Scripts create dirs automatically, check write permissions
- **Notebook cells**: Some have saved outputs from previous runs

## Key Files to Understand
- `docs/main.js` - Navigation and view management
- `docs/js/data.js` - Embedded dashboard data and utilities
- `docs/Scripts/marketing_analytics.py` - Representative ETL + ML workflow
- `python.py` - Comprehensive ML examples and model snippets

This codebase demonstrates **end-to-end analytics** from data generation through web presentation, emphasizing modular architecture and domain separation.
