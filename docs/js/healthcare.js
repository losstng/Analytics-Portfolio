/**
 * Healthcare analytics domain module.
 * Contains templates, chart configuration, and data specific to healthcare projects.
 */

/**
 * Healthcare project configuration
 */
export const healthcareProjects = [
  {
    title: 'Patient Readmission Prediction',
    description: 'Model readmission risk with logistic regression.',
    link: '../Scripts/patient_readmission_model.py',
    tags: ['Classification', 'Healthcare'],
    chartKey: 'Healthcare'
  }
];

/**
 * Create chart configuration for Healthcare category
 */
export function createHealthcareChart(data) {
  const labels = Object.keys(data.health_metrics);
  const values = labels.map((k) => data.health_metrics[k]);

  return {
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
}
