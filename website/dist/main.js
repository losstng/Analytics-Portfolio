"use strict";
const projects = {
    Operations: [
        {
            title: 'Supply Chain Optimization',
            description: 'Optimize inventory using forecasting and EOQ.',
            link: '../Scripts/supply_chain_optimization.py'
        }
    ],
    Marketing: [
        {
            title: 'Customer Segmentation & A/B Testing',
            description: 'Cluster customers and evaluate campaign lift.',
            link: '../Scripts/marketing_analytics.py'
        }
    ],
    Healthcare: [
        {
            title: 'Patient Readmission Prediction',
            description: 'Model readmission risk with logistic regression.',
            link: '../Scripts/patient_readmission_model.py'
        }
    ],
    Finance: [
        {
            title: 'Stock Market Analysis',
            description: 'Forecast next-day prices for NVDA.',
            link: '../Scripts/stock_market_analysis.py'
        }
    ]
};
function render(category) {
    const content = document.getElementById('content');
    if (!content)
        return;
    const list = projects[category] || [];
    content.innerHTML = `<h2>${category}</h2>`;
    const ul = document.createElement('ul');
    list.forEach(p => {
        const li = document.createElement('li');
        li.innerHTML = `<strong>${p.title}</strong> - ${p.description} (<a href="${p.link}">code</a>)`;
        ul.appendChild(li);
    });
    content.appendChild(ul);
}
function attachNavHandlers() {
    document.querySelectorAll('#nav a').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            e.preventDefault();
            const target = e.target;
            const category = target.getAttribute('data-category');
            if (category) {
                render(category);
            }
        });
    });
}
document.addEventListener('DOMContentLoaded', attachNavHandlers);
