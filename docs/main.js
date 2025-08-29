/**
 * Main application logic for the analytics portfolio website.
 * This file has been refactored to separate concerns:
 * - Data and configuration moved to js/data.js
 * - HTML templates moved to js/templates.js
 * - Chart rendering moved to js/charts.js
 * 
 * Updated to support CV landing page with domain dropdown navigation.
 */

import { iconFor } from './js/data.js';
import { projects } from './js/templates.js';
import { renderChart } from './js/charts.js';

// State management
let currentView = 'cv'; // 'cv' or 'portfolio'
let currentCategory = null;

/**
 * Show the CV landing page and hide portfolio section
 */
function showCVLanding() {
  currentView = 'cv';
  currentCategory = null;
  
  const cvSection = document.getElementById('cv-landing');
  const portfolioSection = document.getElementById('portfolio-section');
  const notebookBtn = document.getElementById('view-notebook-btn');
  
  if (cvSection) cvSection.removeAttribute('hidden');
  if (portfolioSection) portfolioSection.setAttribute('hidden', '');
  if (notebookBtn) notebookBtn.setAttribute('hidden', '');
  
  // Set active state
  setActive(null);
  
  // Close navigation dropdown if open
  closeNavDropdown();
  
  // Update URL without triggering hashchange
  history.replaceState(null, '', '#');
  
}

/**
 * Show the portfolio section and hide CV landing
 */
function showPortfolio(category) {
  currentView = 'portfolio';
  currentCategory = category;
  
  const cvSection = document.getElementById('cv-landing');
  const portfolioSection = document.getElementById('portfolio-section');
  const domainTitle = document.getElementById('domain-title');
  const notebookBtn = document.getElementById('view-notebook-btn');
  
  if (cvSection) cvSection.setAttribute('hidden', '');
  if (portfolioSection) portfolioSection.removeAttribute('hidden');
  if (domainTitle) domainTitle.textContent = `${category} Analytics Portfolio`;
  
  // Handle notebook button
  if (notebookBtn) {
    const notebookLinks = {
      'Marketing': '../Notebooks/marketing_analytics.ipynb',
      'Finance': '../Notebooks/Master.ipynb', 
      'Healthcare': '../Notebooks/Master.ipynb',
      'Operations': '../Notebooks/Master.ipynb'
    };
    
    const notebookLink = notebookLinks[category];
    if (notebookLink) {
      notebookBtn.href = notebookLink;
      notebookBtn.removeAttribute('hidden');
    } else {
      notebookBtn.setAttribute('hidden', '');
    }
  }
  
  // Set active state
  setActive(category);
  
  // Close navigation dropdown if open
  closeNavDropdown();
  
  // Render the category
  render(category);
  
  // Update URL
  history.replaceState(null, '', `#${category}`);
}

/**
 * Update the active state of the navigation links to reflect the currently
 * selected category. This sets aria-current appropriately for accessibility.
 * @param {string} category
 */
function setActive(category) {
  // Update navigation button state
  const navBtn = document.getElementById('nav-portfolio-btn');
  const cvBtn = document.getElementById('cv-nav-btn');

  if (category) {
    if (navBtn) {
      navBtn.classList.add('active');
      navBtn.setAttribute('aria-current', 'page');
      navBtn.setAttribute('aria-pressed', 'true');
    }
    if (cvBtn) {
      cvBtn.classList.remove('active');
      cvBtn.removeAttribute('aria-current');
      cvBtn.setAttribute('aria-pressed', 'false');
    }
  } else {
    if (navBtn) {
      navBtn.classList.remove('active');
      navBtn.removeAttribute('aria-current');
      navBtn.setAttribute('aria-pressed', 'false');
    }
    if (cvBtn) {
      cvBtn.classList.add('active');
      cvBtn.setAttribute('aria-current', 'page');
      cvBtn.setAttribute('aria-pressed', 'true');
    }
  }
}


/**
 * Render the list of projects for a given category. Cards can be charts
 * or embedded content. After inserting the markup, we render charts
 * for any cards of type 'chart'.
 * @param {string} category
 */
function render(category) {
  const grid = document.getElementById('grid');
  if (!grid) return;
  const list = projects[category] || [];
  setActive(category);

  // Generate card HTML. Chart cards get a unique canvas id.
  const cards = list
    .map((p, idx) => {
      const type = p.type || 'chart';
      const tags = (p.tags || [])
        .map((t) => `<span class="badge">${t}</span>`)
        .join('');
      const canvasId = `chart-${category}-${idx}`;
      const maybeChart = type === 'chart'
        ? `<div class="chart-container"><canvas id="${canvasId}" aria-label="Dashboard chart for ${category}" role="img"></canvas></div>`
        : '';
      const maybeEmbed = type !== 'chart'
        ? `<div class="embed-container">${p.contentHtml || ''}</div>`
        : '';
      const maybeActions = p.link
        ? `<div class="actions"><a class="btn" href="${p.link}" target="_blank" rel="noopener">View Code</a></div>`
        : '';
      return `
        <article class="card">
          <header class="card-header">
            <div class="icon">${iconFor(category)}</div>
            <div>
              <h3>${p.title}</h3>
              <div class="meta">${tags}</div>
            </div>
          </header>
          <p>${p.description || ''}</p>
          ${maybeChart}
          ${maybeEmbed}
          ${maybeActions}
        </article>
      `;
    })
    .join('');

  grid.innerHTML = cards || `<p class="lead">No projects yet in <strong>${category}</strong>.</p>`;

  // Render all charts for this category after the DOM has been updated.
  requestAnimationFrame(() => {
    list.forEach((p, idx) => {
      const type = p.type || 'chart';
      if (type === 'chart') {
        const canvasId = `chart-${category}-${idx}`;
        renderChart(category, canvasId).catch((err) => console.error(err));
      }
    });
  });
}

/**
 * Initialize the application and attach all event handlers
 */
function initializeApp() {
  // Header navigation portfolio dropdown button
  const navPortfolioBtn = document.getElementById('nav-portfolio-btn');
  if (navPortfolioBtn) {
    navPortfolioBtn.addEventListener('click', toggleNavDropdown);
  }
  
  // CV navigation button
  const cvNavBtn = document.getElementById('cv-nav-btn');
  if (cvNavBtn) {
    cvNavBtn.addEventListener('click', showCVLanding);
  }
  
  // Domain cards in navigation dropdown
  document.querySelectorAll('.nav-domain-card').forEach((card) => {
    card.addEventListener('click', (e) => {
      const category = e.currentTarget.getAttribute('data-category');
      if (category) {
        showPortfolio(category);
      }
    });
  });
  
  // Back to CV button
  const backBtn = document.getElementById('back-to-cv');
  if (backBtn) {
    backBtn.addEventListener('click', showCVLanding);
  }
  
  // Handle browser navigation (back/forward and hash changes)
  window.addEventListener('popstate', handleNavigation);
  window.addEventListener('hashchange', handleNavigation);
  
  // Initialize based on current URL
  requestAnimationFrame(() => {
    handleNavigation();
  });
}

/**
 * Toggle the navigation dropdown
 */
function toggleNavDropdown() {
  const button = document.getElementById('nav-portfolio-btn');
  const dropdown = document.getElementById('nav-domain-dropdown');
  
  if (!button || !dropdown) return;
  
  const isExpanded = button.getAttribute('aria-expanded') === 'true';
  
  if (isExpanded) {
    closeNavDropdown();
  } else {
    openNavDropdown();
  }
}

/**
 * Open the navigation dropdown
 */
function openNavDropdown() {
  const button = document.getElementById('nav-portfolio-btn');
  const dropdown = document.getElementById('nav-domain-dropdown');
  
  if (!button || !dropdown) return;
  
  button.setAttribute('aria-expanded', 'true');
  dropdown.removeAttribute('hidden');
  
  // Add click-outside listener
  setTimeout(() => {
    document.addEventListener('click', handleNavClickOutside);
  }, 0);
}

/**
 * Close the navigation dropdown
 */
function closeNavDropdown() {
  const button = document.getElementById('nav-portfolio-btn');
  const dropdown = document.getElementById('nav-domain-dropdown');
  
  if (!button || !dropdown) return;
  
  button.setAttribute('aria-expanded', 'false');
  dropdown.setAttribute('hidden', '');
  
  // Remove click-outside listener
  document.removeEventListener('click', handleNavClickOutside);
}

/**
 * Handle clicks outside the navigation dropdown to close it
 */
function handleNavClickOutside(event) {
  const dropdown = document.getElementById('nav-domain-dropdown');
  const button = document.getElementById('nav-portfolio-btn');
  
  if (!dropdown || !button) return;
  
  if (!dropdown.contains(event.target) && !button.contains(event.target)) {
    closeNavDropdown();
  }
}

/**
 * Handle navigation based on URL hash
 */
function handleNavigation() {
  const hash = (location.hash || '').replace('#', '');
  if (!hash || hash === 'cv') {
    showCVLanding();
  } else if (['Marketing', 'Finance', 'Healthcare', 'Operations'].includes(hash)) {
    showPortfolio(hash);
  } else {
    // Unknown hash, default to CV
    showCVLanding();
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);