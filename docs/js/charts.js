/**
 * Unified chart rendering system using domain-specific modules.
 * This file coordinates chart rendering across all domains.
 */

import { loadDashboardData, chartInstances } from './data.js';
import { createOperationsChart } from './operations.js';
import { createMarketingChart } from './marketing.js';
import { createHealthcareChart } from './healthcare.js';
import { createFinanceChart } from './finance.js';

/**
 * Chart creator functions mapped by category
 */
const chartCreators = {
  Operations: createOperationsChart,
  Marketing: createMarketingChart,
  Healthcare: createHealthcareChart,
  Finance: createFinanceChart,
};

/**
 * Draw the appropriate chart inside the provided canvas for a given category.
 * If a chart for that canvas already exists it is destroyed before
 * creating a new instance. The function uses the precomputed JSON
 * summaries to populate chart datasets.
 * @param {string} category
 * @param {string} canvasId
 */
export async function renderChart(category, canvasId) {
  const data = await loadDashboardData();
  const canvas = document.getElementById(canvasId);
  if (!(canvas && canvas.getContext)) return;

  // Clean up any existing chart instance for this canvas
  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
  }

  // Get the chart creator for this category
  const createChart = chartCreators[category];
  if (!createChart) {
    console.warn(`No chart creator found for category: ${category}`);
    return;
  }

  const ctx = canvas.getContext('2d');
  const config = createChart(data);
  
  if (config) {
    chartInstances[canvasId] = new Chart(ctx, config);
  }
}
