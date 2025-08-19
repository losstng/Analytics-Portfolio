# Portfolio Website Integration Plan: marketing_analytics.ipynb

This document outlines a coherent plan to showcase the data cleaning process and visualizations from `marketing_analytics.ipynb` on the portfolio website.

## 1. Data Cleaning Showcase

### a. Summarize Cleaning Steps
- Extract markdown explanations and code snippets related to data cleaning from the notebook.
- Present these as formatted sections on the project page (e.g., using `<pre>` or `<code>` blocks for code, and paragraphs for explanations).

### b. Before-and-After Table
- Export a sample (e.g., 5 rows) of the raw data before cleaning as an image or HTML table.
- Export a sample (e.g., the same 5 rows) of the cleaned data as an image or HTML table.
- Embed both samples side-by-side or sequentially on the project page to visually demonstrate the effect of cleaning.

### c. Optional: Interactive Table
- Optionally, provide downloadable CSVs or use a JavaScript table library for interactive exploration.

## 2. Visualization Showcase

### a. Export Key Plots
- Save each important plot from the notebook as a PNG or SVG file (e.g., using `plt.savefig('website/assets/plot1.png')`).
- Place these images in a dedicated folder (e.g., `website/assets/`).

### b. Embed in Project Page
- In `website/projects/marketing_analytics.html`, embed the images using `<img>` tags with appropriate captions.
- Optionally, add short explanations for each plot.

### c. Optional: Interactive Visuals
- For interactive charts, export cleaned data as CSV and use JavaScript libraries (e.g., Plotly.js, Chart.js) to recreate the plots on the website.

## 3. Project Page Structure Example

1. **Introduction**: Brief overview of the project and objectives.
2. **Data Cleaning**:
    - Cleaning steps (text + code)
    - Before-and-after data samples
3. **Analysis & Visualizations**:
    - Key plots with captions
    - (Optional) Interactive charts
4. **Conclusion**: Key findings and insights.

---

**Next Steps:**
- Extract and format the cleaning steps and code.
- Export before/after data samples.
- Export and organize plot images.
- Build the HTML project page using the above structure.
