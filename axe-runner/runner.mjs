#!/usr/bin/env node
/**
 * axe-core-runner — WCAG accessibility audit for ZSEL K8s CronJob
 * Runs axe-core via Playwright against a list of URLs, posts results to Sentinel API.
 *
 * Usage:
 *   node runner.mjs --urls <url1,url2> --standard WCAG21AA \
 *     --post-to <api-url> --token-from-secret /etc/accessibility/token
 */

import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import process from "node:process";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright-core");
const { default: axe } = require("axe-core");

const { values: args } = parseArgs({
  options: {
    urls:              { type: "string" },
    standard:          { type: "string", default: "WCAG21AA" },
    "post-to":         { type: "string" },
    "token-from-secret": { type: "string" },
  },
});

if (!args.urls) {
  console.error("ERROR: --urls jest wymagane");
  process.exit(1);
}

const urls = args.urls.split(",").map((u) => u.trim()).filter(Boolean);
const standard = args.standard;
const postTo = args["post-to"];
const tokenPath = args["token-from-secret"];
const token = tokenPath ? readFileSync(tokenPath, "utf8").trim() : null;

const runAxe = async (url) => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || "/usr/bin/chromium",
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--headless=new",
    ],
  });
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.addScriptTag({ content: axe.source });
    const results = await page.evaluate(async (tags) => {
      return await window.axe.run({ runOnly: { type: "tag", values: tags } });
    }, [standard.toLowerCase()]);

    return {
      url,
      standard,
      violations: results.violations.length,
      passes: results.passes.length,
      incomplete: results.incomplete.length,
      details: results.violations.map((v) => ({
        id: v.id,
        impact: v.impact,
        description: v.description,
        nodes: v.nodes.length,
      })),
    };
  } finally {
    await browser.close();
  }
};

const main = async () => {
  const ts = new Date().toISOString();
  const report = {
    timestamp: ts,
    standard,
    results: [],
    summary: { total_violations: 0, urls_checked: urls.length, urls_failed: 0 },
  };

  for (const url of urls) {
    try {
      console.log(`[${ts}] Audyt: ${url}`);
      const result = await runAxe(url);
      report.results.push(result);
      report.summary.total_violations += result.violations;
      console.log(`  OK — naruszenia: ${result.violations}, poprawne: ${result.passes}`);
    } catch (err) {
      console.error(`  BŁĄD dla ${url}: ${err.message}`);
      report.results.push({ url, error: err.message });
      report.summary.urls_failed += 1;
    }
  }

  console.log(JSON.stringify({ summary: report.summary }, null, 2));

  if (postTo) {
    const headers = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;

    const response = await fetch(postTo, {
      method: "POST",
      headers,
      body: JSON.stringify(report),
    });
    if (!response.ok) {
      console.error(`POST ${postTo} nieudany: ${response.status}`);
      process.exit(2);
    }
    console.log(`OK Wyniki wysłane do ${postTo}`);
  }
};

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
