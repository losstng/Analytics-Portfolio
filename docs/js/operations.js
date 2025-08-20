/**
 * Operations analytics domain module.
 * Contains templates, chart configuration, and data specific to operations projects.
 */

/**
 * Operations project configuration
 */
export const operationsProjects = [
  {
    title: 'Supply Chain Optimization',
    description: 'Optimize inventory using forecasting and EOQ.',
    link: '../Scripts/supply_chain_optimization.py',
    tags: ['Forecasting', 'EOQ'],
    chartKey: 'Operations'
  }
];

/**
 * Create chart configuration for Operations category
 */
export function createOperationsChart(data) {
  const labels = Object.keys(data.supply_forecasts);
  const forecastVals = labels.map((k) => data.supply_forecasts[k]);
  const eoqVals = labels.map((k) => data.supply_eoqs[k]);

  return {
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
}
