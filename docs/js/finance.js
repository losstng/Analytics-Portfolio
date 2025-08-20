/**
 * Finance analytics domain module.
 * Contains templates, chart configuration, and data specific to finance projects.
 */

/**
 * Finance project configuration
 */
export const financeProjects = [
  {
    title: 'Stock Market Analysis',
    description: 'Forecast next-day prices for NVDA.',
    link: '../Scripts/stock_market_analysis.py',
    tags: ['Time Series', 'Finance'],
    chartKey: 'Finance'
  }
];

/**
 * Create chart configuration for Finance category
 */
export function createFinanceChart(data) {
  const labels = data.finance_dates;
  const values = data.finance_close;

  return {
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
}
