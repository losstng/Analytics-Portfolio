/**
 * Project configuration aggregator.
 * Imports and combines projects from all domain modules.
 */

import { operationsProjects } from './operations.js';
import { marketingProjects } from './marketing.js';
import { healthcareProjects } from './healthcare.js';
import { financeProjects } from './finance.js';

/**
 * Complete project configuration for all domains.
 * Each domain module handles its own templates and chart configurations.
 */
export const projects = {
  Operations: operationsProjects,
  Marketing: marketingProjects,
  Healthcare: healthcareProjects,
  Finance: financeProjects
};