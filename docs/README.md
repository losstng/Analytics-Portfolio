# Portfolio Website (docs)

This folder contains the refactored analytics portfolio site used for GitHub Pages or local previews. The site follows a domain-based modular structure so each business area (Marketing, Operations, Healthcare, Finance) keeps templates, projects, and chart logic together.

## Quick-start — view locally

1. Serve the `docs/` directory over HTTP (recommended):

```bash
cd docs
python3 -m http.server 8000

# then open in your browser:
# http://127.0.0.1:8000/
```

Note: opening `index.html` directly via the file:// protocol can break ES module imports and make the page appear static (no interactivity). If tabs like Operations / Marketing don't respond, try the HTTP server above and check the browser console for errors.

## File structure (summary)

```
docs/
├── index.html          # Main HTML page (loads the modular JS)
├── main.js             # Small bootstrap renderer (imports domain projects)
├── styles.css          # Site styles
├── notebooks/          # Jupyter notebooks and markdown files
│   ├── Master.ipynb    # Master notebook
│   ├── Showcase.ipynb  # Showcase notebook
│   ├── marketing_analytics.ipynb  # Marketing analytics notebook
│   └── marketing_analytics.md     # Marketing analytics documentation
├── data/               # Generated and processed data files
│   ├── marketing_data.csv         # Primary marketing dataset
│   └── marketing_data_2.csv       # Secondary marketing dataset
├── js/
│   ├── data.js        # Embedded dashboard data, icons, utilities
│   ├── templates.js   # Aggregates domain project arrays
│   ├── charts.js      # Chart coordinator (delegates to domain creators)
│   ├── marketing.js   # Marketing domain: templates, projects, createMarketingChart()
│   ├── operations.js  # Operations domain: templates, projects, createOperationsChart()
│   ├── healthcare.js  # Healthcare domain: templates, projects, createHealthcareChart()
│   └── finance.js     # Finance domain: templates, projects, createFinanceChart()
└── assets/            # Static assets (images, exported HTML, CSVs)
    └── marketing_analytics/  # Marketing analytics specific assets
```

## How it works (high level)

- `index.html` loads `main.js` as an ES module.
- `main.js` imports `projects` from `js/templates.js` and `renderChart` from `js/charts.js` and is responsible for navigation and rendering the project grid.
- `js/templates.js` aggregates exported project arrays from domain modules (marketing/operations/healthcare/finance).
- Each domain module exports a projects array and a `create*Chart(data)` function used by `js/charts.js` to build Chart.js configs.
- `js/data.js` contains embedded `DASHBOARD_DATA` used by charts so the site can work without any backend.

## Troubleshooting

- If the navigation links are visible but clicking them does nothing:
	- Make sure you opened the site over HTTP (see Quick-start). file:// often prevents module loading.
	- Open Developer Tools → Console and look for errors. Common problems:
		- "Failed to load module script" or MIME type errors — means modules weren't served correctly.
		- Uncaught exceptions during import — copy the stack trace here and I can debug it.

- If images or iframe HTML don't load, check `assets/marketing_analytics/*` exists and paths are correct relative to `docs/`.

## Adding a new domain (short)

1. Create `js/<domain>.js` exporting two things:
	 - `export const <domain>Projects = [ /* project objects */ ];`
	 - `export function create<Domain>Chart(data) { return Chart.js config; }`
2. Import the projects in `js/templates.js` and add to the exported `projects` object.
3. Import the chart creator in `js/charts.js` and add it to the `chartCreators` map.
4. Optionally add a nav link in `index.html`.

## Contributor notes

- Keep domain modules focused: templates, projects meta, and chart builders belong together.
- Prefer `type: 'embed'` for iframe/embed content and `type: 'chart'` for Chart.js-driven cards.
- When editing `js/data.js`, keep the embedded JSON small (only what's needed) to avoid slowing page load.

If you'd like, I can add a small warning banner to the page that appears when modules fail to load (helpful for demos). Reply if you want that added and I'll implement it.
