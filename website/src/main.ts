interface Project {
  title: string;
  description: string;
  link: string;
}

const projects: Record<string, Project[]> = {
  Operations: [
    {
      title: 'Supply Chain Optimization',
      description: 'Optimize inventory using forecasting and EOQ.',
      link: 'projects/supply_chain_optimization.html'
    }
  ],
  Marketing: [
    {
      title: 'Customer Segmentation & A/B Testing',
      description: 'Cluster customers and evaluate campaign lift.',
      link: 'projects/marketing_analytics.html'
    }
  ],
  Healthcare: [
    {
      title: 'Patient Readmission Prediction',
      description: 'Model readmission risk with logistic regression.',
      link: 'projects/patient_readmission_model.html'
    }
  ],
  Finance: [
    {
      title: 'Stock Market Analysis',
      description: 'Forecast next-day prices for NVDA.',
      link: 'projects/stock_market_analysis.html'
    }
  ]
};

function render(category: string): void {
  const content = document.getElementById('content');
  if (!content) return;
  const list = projects[category] || [];
  content.innerHTML = `<h2>${category}</h2>`;
  const ul = document.createElement('ul');
  list.forEach(p => {
    const li = document.createElement('li');
    li.innerHTML = `<strong>${p.title}</strong> - ${p.description} (<a href="${p.link}">details</a>)`;
    ul.appendChild(li);
  });
  content.appendChild(ul);
}

function attachNavHandlers(): void {
  document.querySelectorAll('#nav a').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      e.preventDefault();
      const target = e.target as HTMLElement;
      const category = target.getAttribute('data-category');
      if (category) {
        render(category);
      }
    });
  });
}

document.addEventListener('DOMContentLoaded', attachNavHandlers);
