interface Project {
  title: string;
  description: string;
  link: string;
  tags?: string[];
  icon?: string; // optional per-project override
}

const projects: Record<string, Project[]> = {
  Operations: [
    {
      title: 'Supply Chain Optimization',
      description: 'Optimize inventory using forecasting and EOQ.',
      link: '../Scripts/supply_chain_optimization.py',
      tags: ['EOQ', 'Forecasting']
    }
  ],
  Marketing: [
    {
      title: 'Customer Segmentation & A/B Testing',
      description: 'Cluster customers and evaluate campaign lift.',
      link: '../Scripts/marketing_analytics.py',
      tags: ['Clustering', 'Experimentation']
    }
  ],
  Healthcare: [
    {
      title: 'Patient Readmission Prediction',
      description: 'Model readmission risk with logistic regression.',
      link: '../Scripts/patient_readmission_model.py',
      tags: ['Logistic', 'Risk']
    }
  ],
  Finance: [
    {
      title: 'Stock Market Analysis',
      description: 'Forecast next-day prices for NVDA.',
      link: '../Scripts/stock_market_analysis.py',
      tags: ['Time Series', 'NVDA']
    }
  ]
};

/* --- Minimal, crisp inline icons (no external libs) --- */
const categoryIcons: Record<string, string> = {
  Operations: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z" stroke="currentColor" stroke-width="1.6"/>
      <path d="M4 13h2l1 2 2 1v2l3 2 3-2v-2l2-1 1-2h2l1-2-1-2h-2l-1-2-2-1V4l-3-2-3 2v2l-2 1-1 2H4l-1 2 1 2Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
    </svg>`,
  Marketing: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M3 10v4m0-4 12-5v14l-12-5Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
      <path d="M21 9v6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M18 10v4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
    </svg>`,
  Healthcare: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 22c5-4 8-7.5 8-12a8 8 0 1 0-16 0c0 4.5 3 8 8 12Z" stroke="currentColor" stroke-width="1.6"/>
      <path d="M8 12h3l1.2-3.2L14 14h2" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`,
  Finance: `
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 19h16" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
      <path d="M6 15l3-3 3 2 5-6 1 1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>`
};

function iconFor(category: string, project?: Project): string {
  if (project?.icon) return project.icon;
  return categoryIcons[category] || categoryIcons.Operations;
}

function setActive(category: string): void {
  document.querySelectorAll('#nav a').forEach(a => {
    const el = a as HTMLAnchorElement;
    const isActive = el.getAttribute('data-category') === category;
    if (isActive) {
      el.setAttribute('aria-current', 'page');
    } else {
      el.removeAttribute('aria-current');
    }
  });
}

function render(category: string): void {
  const grid = document.getElementById('grid');
  if (!grid) return;

  const list = projects[category] || [];
  setActive(category);

  const cards = list.map(p => {
    const tags = (p.tags || []).map(t => `<span class="badge">${t}</span>`).join('');
    return `
      <article class="card">
        <header class="card-header">
          <div class="icon">${iconFor(category, p)}</div>
          <div>
            <h3>${p.title}</h3>
            <div class="meta">${tags}</div>
          </div>
        </header>
        <p>${p.description}</p>
        <div class="actions">
          <a class="btn" href="${p.link}" target="_blank" rel="noopener">View Code</a>
        </div>
      </article>
    `;
  }).join('');

  grid.innerHTML = cards || `<p class="lead">No projects yet in <strong>${category}</strong>.</p>`;
}

/* Handle nav clicks + deep-link via hash */
function attachNavHandlers(): void {
  document.querySelectorAll('#nav a').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      e.preventDefault();
      const target = e.currentTarget as HTMLElement;
      const category = target.getAttribute('data-category');
      if (category) {
        location.hash = category; // persist selection
        render(category);
      }
    });
  });

  // load from hash or default
  const initial = (location.hash || '#Operations').replace('#', '');
  render(initial);

  // allow manual hash change
  window.addEventListener('hashchange', () => {
    const cat = (location.hash || '#Operations').replace('#', '');
    render(cat);
  });
}

document.addEventListener('DOMContentLoaded', attachNavHandlers);
