/**
 * Main application logic for the analytics portfolio website.
 * This file has been refactored to separate concerns:
 * - Data and configuration moved to js/data.js
 * - HTML templates moved to js/templates.js
 * - Chart rendering moved to js/charts.js
 */

import { iconFor } from './js/data.js';
import { projects } from './js/templates.js';
import { renderChart } from './js/charts.js';

/**
 * Update the active state of the navigation links to reflect the currently
 * selected category. This sets aria-current appropriately for accessibility.
 * @param {string} category
 */
function setActive(category) {
  document.querySelectorAll('#nav a').forEach((a) => {
    const el = /** @type {HTMLAnchorElement} */ (a);
    const isActive = el.getAttribute('data-category') === category;
    if (isActive) {
      el.setAttribute('aria-current', 'page');
    } else {
      el.removeAttribute('aria-current');
    }
  });
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
 * Attach click handlers to the navigation items and handle deep-linking
 * via URL hash. If the user navigates directly to /#Finance, that
 * category will be selected on page load. Subsequent hash changes
 * trigger re-renders.
 */
function attachNavHandlers() {
  document.querySelectorAll('#nav a').forEach((anchor) => {
    anchor.addEventListener('click', (e) => {
      e.preventDefault();
      const target = /** @type {HTMLElement} */ (e.currentTarget);
      const category = target.getAttribute('data-category');
      if (category) {
        location.hash = category;
        render(category);
      }
    });
  });

  // load from hash or default
  const initial = (location.hash || '#Operations').replace('#', '');
  render(initial);

  // update on manual hash change
  window.addEventListener('hashchange', () => {
    const cat = (location.hash || '#Operations').replace('#', '');
    render(cat);
  });
}

document.addEventListener('DOMContentLoaded', attachNavHandlers);