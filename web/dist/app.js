/**
 * NexusWatch AI — Investigation Console Application Logic
 * Next-Gen Graph Fraud Intelligence on TigerGraph
 */

// Application State
const state = {
  cases: [],
  selectedId: null,
  activeFilter: 'all',
  searchQuery: '',
  activeTab: 'overview',
  theme: localStorage.getItem('nw_theme') || 'dark'
};

// Utilities
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, m => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[m]));

const money = (num) => new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2
}).format(num || 0);

const formatPattern = (p) => {
  if (!p || p === 'none') return 'Legitimate Pattern Baseline';
  return p.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
};

// SVG Graph Generator with high contrast and readable labels
function generateSvgSubgraph(item) {
  const isFraud = item.case.verdict === 'fraud';
  const cardId = item.trigger.card_id || 'CARD-UNKNOWN';
  const txnId = item.trigger.flagged_txn_id || 'TXN-FLAGGED';
  const device = item.case.connected_device_profiles[0] 
    ? item.case.connected_device_profiles[0].split('|')[0].trim().slice(0, 16) 
    : 'Standard Browser';
  const priorCase = item.case.similar_prior_cases[0] || 'Base-R1';
  const policyRule = item.next_best_actions.final[0]?.reason.split(':')[0] || 'R1-Gate';

  // Node coordinates (SVG 620 x 240)
  const nodes = [
    { id: 'card', label: cardId, sub: 'Card Account', x: 85, y: 120, color: '#00f5ff', r: 30 },
    { id: 'txn', label: `TXN ${txnId}`, sub: money(item.case.exposure_usd), x: 235, y: 65, color: isFraud ? '#ff3366' : '#10b981', r: 32 },
    { id: 'device', label: device, sub: 'Device Fingerprint', x: 235, y: 180, color: '#c084fc', r: 28 },
    { id: 'memory', label: priorCase, sub: 'Prior Cluster', x: 410, y: 70, color: '#fbbf24', r: 28 },
    { id: 'policy', label: policyRule, sub: item.case.verdict.toUpperCase(), x: 535, y: 120, color: isFraud ? '#ff3366' : '#10b981', r: 30 }
  ];

  // Edges definition
  const edges = [
    { from: nodes[0], to: nodes[1], label: 'transacted' },
    { from: nodes[0], to: nodes[2], label: 'fingerprint' },
    { from: nodes[1], to: nodes[3], label: 'resembles' },
    { from: nodes[3], to: nodes[4], label: 'evaluates' },
    { from: nodes[2], to: nodes[4], label: 'verifies' }
  ];

  let svg = `
  <svg class="subgraph-svg" viewBox="0 0 620 240" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#475569" />
        <stop offset="100%" stop-color="#94a3b8" />
      </linearGradient>
      <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>

    <!-- Grid lines -->
    <pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">
      <path d="M 24 0 L 0 0 0 24" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
    </pattern>
    <rect width="100%" height="100%" fill="url(#grid)" />

    <!-- Connecting Edges -->
    <g class="edges">
  `;

  edges.forEach(e => {
    svg += `
      <line x1="${e.from.x}" y1="${e.from.y}" x2="${e.to.x}" y2="${e.to.y}" 
            stroke="url(#edgeGrad)" stroke-width="2" stroke-dasharray="5,5" />
      <text x="${(e.from.x + e.to.x)/2}" y="${(e.from.y + e.to.y)/2 - 6}" 
            fill="#94a3b8" font-size="10" font-weight="600" font-family="'JetBrains Mono', monospace" text-anchor="middle">
        ${e.label}
      </text>
    `;
  });

  svg += `</g><g class="nodes">`;

  // Draw Nodes
  nodes.forEach(n => {
    svg += `
      <g class="graph-node" transform="translate(${n.x}, ${n.y})">
        <circle r="${n.r}" fill="#0b1120" stroke="${n.color}" stroke-width="2.5" filter="url(#glow)" />
        <text y="-2" fill="#ffffff" font-size="10" font-weight="800" font-family="'JetBrains Mono', monospace" text-anchor="middle">
          ${esc(n.label)}
        </text>
        <text y="12" fill="#cbd5e1" font-size="8.5" font-weight="600" font-family="'JetBrains Mono', monospace" text-anchor="middle">
          ${esc(n.sub)}
        </text>
      </g>
    `;
  });

  svg += `</g></svg>`;
  return svg;
}

// Compute filtered list
function getVisibleCases() {
  const q = state.searchQuery.toLowerCase().trim();
  return state.cases.filter(c => {
    const verdictMatch = 
      state.activeFilter === 'all' ||
      (state.activeFilter === 'fraud' && c.case.verdict === 'fraud') ||
      (state.activeFilter === 'legitimate' && c.case.verdict === 'legitimate') ||
      (state.activeFilter === 'high-risk' && c.case.exposure_usd >= 500);

    if (!verdictMatch) return false;
    if (!q) return true;

    const searchBlob = [
      c.case_id,
      c.trigger.card_id,
      c.trigger.flagged_txn_id,
      c.case.pattern,
      c.case.summary,
      c.sar.reason
    ].join(' ').toLowerCase();

    return searchBlob.includes(q);
  });
}

// Render Case Queue List
function renderQueue() {
  const visible = getVisibleCases();
  const queueEl = $('#caseQueueList');
  $('#filteredCountBadge').textContent = `${visible.length} CASE${visible.length === 1 ? '' : 'S'}`;

  if (visible.length === 0) {
    queueEl.innerHTML = `
      <div style="padding: 2.5rem 1.5rem; text-align: center; color: var(--text-muted); font-family: var(--font-mono); font-size: 0.85rem;">
        No cases match your filter criteria.
      </div>
    `;
    return;
  }

  queueEl.innerHTML = visible.map(c => {
    const isSelected = c.case_id === state.selectedId;
    const isFraud = c.case.verdict === 'fraud';
    const riskPct = Math.round((c.case.fraud_probability || 0) * 100);

    return `
      <div class="case-item ${isSelected ? 'active' : ''} ${isFraud ? 'fraud-case' : ''}" 
           data-id="${c.case_id}" 
           role="option" 
           aria-selected="${isSelected}"
           tabindex="0">
        <span class="case-status-pip ${isFraud ? 'fraud' : ''}"></span>
        <div class="case-info">
          <div class="case-row-top">
            <span class="case-id-text">${c.case_id}</span>
            <span class="case-pattern-tag">${esc(formatPattern(c.case.pattern))}</span>
          </div>
          <div class="case-row-bottom">
            ${esc(c.trigger.card_id)} · TXN ${esc(c.trigger.flagged_txn_id)}
          </div>
        </div>
        <div class="case-meta-right">
          <span class="risk-chip ${isFraud ? 'fraud' : ''}">${riskPct}% Risk</span>
          <span class="case-exp-text">${money(c.case.exposure_usd)}</span>
        </div>
      </div>
    `;
  }).join('');

  // Event Listeners for queue rows
  $$('.case-item').forEach(el => {
    const handleSelect = () => {
      state.selectedId = el.dataset.id;
      renderQueue();
      renderDetail();
    };
    el.addEventListener('click', handleSelect);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleSelect();
      }
    });
  });
}

// Render Case Detail View
function renderDetail() {
  const detailEl = $('#caseDetailPanel');
  const activeCase = state.cases.find(c => c.case_id === state.selectedId) || getVisibleCases()[0];

  if (!activeCase) {
    detailEl.innerHTML = `
      <div class="empty-state" style="padding: 4rem 2rem; text-align: center; color: var(--text-muted);">
        <p>Select a case from the queue to inspect graph evidence.</p>
      </div>
    `;
    return;
  }

  state.selectedId = activeCase.case_id;
  const c = activeCase.case;
  const isFraud = c.verdict === 'fraud';
  const riskPct = Math.round((c.fraud_probability || 0) * 100);

  detailEl.innerHTML = `
    <!-- Header Section -->
    <header class="case-header">
      <div class="case-headline">
        <div class="case-meta-tags">
          <span class="trigger-badge">${esc(activeCase.trigger.type.replace('_', ' '))}</span>
          <span class="case-timestamp">Opened: ${esc(activeCase.trigger.opened_at)}</span>
        </div>
        <h2 class="case-title">${esc(formatPattern(c.pattern))}</h2>
        <div class="case-sub-details">
          <span><strong>Case ID:</strong> ${esc(activeCase.case_id)}</span>
          <span>·</span>
          <span><strong>Card:</strong> ${esc(activeCase.trigger.card_id)}</span>
          <span>·</span>
          <span><strong>Flagged TXN:</strong> ${esc(activeCase.trigger.flagged_txn_id)}</span>
          <span>·</span>
          <span><strong>Customer:</strong> ${esc(activeCase.trigger.customer_id)}</span>
        </div>
      </div>

      <div class="verdict-box">
        <div class="verdict-badge ${isFraud ? 'fraud' : 'legitimate'}">
          ${isFraud ? '🚨 FRAUD CONFIRMED' : '🛡️ CLEARED / BENIGN'} (${riskPct}%)
        </div>
        <div class="sar-status-pill ${activeCase.sar.file ? 'sar-required' : ''}">
          ${activeCase.sar.file ? '⚠️ FinCEN SAR Required' : '✓ No SAR Filing Needed'}
        </div>
      </div>
    </header>

    <!-- Metrics Bar -->
    <div class="metric-strip">
      <div class="strip-item">
        <div class="strip-label">FINANCIAL EXPOSURE</div>
        <div class="strip-val ${isFraud ? 'text-fraud' : 'text-legit'}">${money(c.exposure_usd)}</div>
      </div>
      <div class="strip-item">
        <div class="strip-label">AFFECTED TRANSACTIONS</div>
        <div class="strip-val">${c.affected_txn_ids?.length || 1} TXNs</div>
      </div>
      <div class="strip-item">
        <div class="strip-label">HISTORICAL SIMILAR CASES</div>
        <div class="strip-val">${c.similar_prior_cases?.length || 0} Cases</div>
      </div>
      <div class="strip-item">
        <div class="strip-label">GRAPH WRITEBACK STATUS</div>
        <div class="strip-val text-cyan">${c.written_to_graph ? 'STORED' : 'PENDING'}</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <nav class="detail-tabs" role="tablist">
      <button class="tab-btn ${state.activeTab === 'overview' ? 'active' : ''}" data-tab="overview">
        <span>🧭 Overview & Subgraph</span>
      </button>
      <button class="tab-btn ${state.activeTab === 'policy' ? 'active' : ''}" data-tab="policy">
        <span>🛡️ Policy Execution (R1–R10)</span>
      </button>
      <button class="tab-btn ${state.activeTab === 'provenance' ? 'active' : ''}" data-tab="provenance">
        <span>🔗 Evidence Ledger</span>
      </button>
      <button class="tab-btn ${state.activeTab === 'sar' ? 'active' : ''}" data-tab="sar">
        <span>📑 Regulatory SAR Preview</span>
      </button>
    </nav>

    <!-- TAB 1: OVERVIEW & SUBGRAPH -->
    <div class="tab-pane ${state.activeTab === 'overview' ? 'active' : ''}" id="tab-overview">
      <div class="overview-grid">
        <div>
          <div class="detail-card">
            <div class="card-title-bar">
              <h4><span>📋</span> ANALYST BRIEF & SYNTHESIS</h4>
            </div>
            <p class="summary-text">${esc(c.summary)}</p>
            <div class="trigger-callout">
              <strong>Ingestion Trigger Context:</strong>
              ${esc(activeCase.trigger.text)}
            </div>
          </div>

          <div class="detail-card">
            <div class="card-title-bar">
              <h4><span>⚡</span> AUTOMATED POLICY-CONSTRAINED ACTIONS</h4>
            </div>
            <div class="action-list">
              ${activeCase.next_best_actions.final.map(a => `
                <div class="action-item">
                  <div>
                    <div class="action-name">${esc(a.action)}</div>
                    <div class="action-reason">${esc(a.reason)}</div>
                  </div>
                  <span class="action-route-badge ${a.route}">${esc(a.route.toUpperCase())}</span>
                </div>
              `).join('')}
            </div>
          </div>
        </div>

        <div>
          <div class="detail-card">
            <div class="card-title-bar">
              <h4><span>🕸️</span> TIGERGRAPH EVIDENCE SUBGRAPH</h4>
              <span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--cyan-400); font-weight: 700;">LIVE TOPOLOGY</span>
            </div>
            <div class="subgraph-container">
              ${generateSvgSubgraph(activeCase)}
            </div>
            <div class="graph-legend">
              <div class="legend-item"><span class="legend-dot" style="background:#00f5ff"></span>Card Account</div>
              <div class="legend-item"><span class="legend-dot" style="background:${isFraud ? '#ff3366' : '#10b981'}"></span>Flagged TXN</div>
              <div class="legend-item"><span class="legend-dot" style="background:#c084fc"></span>Device Fingerprint</div>
              <div class="legend-item"><span class="legend-dot" style="background:#fbbf24"></span>Closed Memory</div>
            </div>
          </div>

          <div class="detail-card">
            <div class="card-title-bar">
              <h4><span>📱</span> HARDWARE & NETWORK TOPOLOGY</h4>
            </div>
            <div style="font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-secondary);">
              <div style="margin-bottom: 0.65rem;">
                <strong style="color: var(--text-primary); display:block; margin-bottom: 0.25rem;">Device Fingerprint:</strong>
                <span style="background: var(--bg-main); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-subtle); display: block;">
                  ${esc(c.connected_device_profiles[0] || 'Standard desktop browser session / no anomalous fingerprint.')}
                </span>
              </div>
              <div>
                <strong style="color: var(--text-primary); display:block; margin-bottom: 0.25rem;">Linked Cards / Multi-Account:</strong>
                ${c.connected_card_ids && c.connected_card_ids.length > 0 
                  ? c.connected_card_ids.map(id => `<span style="display:inline-block; margin-right:6px; background:var(--bg-main); padding:3px 8px; border-radius:4px; border:1px solid var(--border-subtle); color:var(--cyan-400);">${esc(id)}</span>`).join('') 
                  : '<span style="color: var(--text-muted);">Isolated card account; no cross-account syndication.</span>'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: POLICY EXECUTION -->
    <div class="tab-pane ${state.activeTab === 'policy' ? 'active' : ''}" id="tab-policy">
      <div class="detail-card">
        <div class="card-title-bar">
          <h4><span>⚖️</span> POLICY EVALUATION & TRANSITION MATRIX</h4>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem; line-height: 1.6;">
          ${esc(activeCase.next_best_actions.what_changed || 'Policy evaluation completed without state mutation.')}
        </p>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
          <div style="background: var(--bg-main); padding: 1.35rem; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
            <div style="font-family: var(--font-mono); font-size: 0.75rem; font-weight: 800; color: var(--text-muted); margin-bottom: 0.85rem; letter-spacing: 0.05em;">INITIAL DETERMINATION</div>
            <div class="action-list">
              ${activeCase.next_best_actions.initial.map(a => `
                <div class="action-item">
                  <div>
                    <div class="action-name">${esc(a.action)}</div>
                    <div class="action-reason">${esc(a.reason)}</div>
                  </div>
                  <span class="action-route-badge ${a.route}">${esc(a.route.toUpperCase())}</span>
                </div>
              `).join('')}
            </div>
          </div>

          <div style="background: var(--bg-main); padding: 1.35rem; border-radius: var(--radius-md); border: 1px solid var(--border-accent);">
            <div style="font-family: var(--font-mono); font-size: 0.75rem; font-weight: 800; color: var(--cyan-400); margin-bottom: 0.85rem; letter-spacing: 0.05em;">FINAL POLICY SETTLEMENT</div>
            <div class="action-list">
              ${activeCase.next_best_actions.final.map(a => `
                <div class="action-item">
                  <div>
                    <div class="action-name">${esc(a.action)}</div>
                    <div class="action-reason">${esc(a.reason)}</div>
                  </div>
                  <span class="action-route-badge ${a.route}">${esc(a.route.toUpperCase())}</span>
                </div>
              `).join('')}
            </div>
          </div>
        </div>

        <div style="margin-top: 1.5rem; padding: 1.15rem; background: rgba(168, 85, 247, 0.1); border-radius: var(--radius-md); border: 1px solid rgba(168, 85, 247, 0.3);">
          <strong style="font-family: var(--font-mono); font-size: 0.8rem; color: #c084fc; display: block; margin-bottom: 0.35rem;">INVESTIGATION STOP REASON:</strong>
          <span style="font-size: 0.88rem; color: var(--text-primary); line-height: 1.55;">${esc(activeCase.stop_reason)}</span>
        </div>
      </div>
    </div>

    <!-- TAB 3: PROVENANCE LEDGER -->
    <div class="tab-pane ${state.activeTab === 'provenance' ? 'active' : ''}" id="tab-provenance">
      <div class="detail-card">
        <div class="card-title-bar">
          <h4><span>⛓️</span> CHAIN-OF-CUSTODY EVIDENCE LEDGER</h4>
          <span style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--cyan-400); font-weight: 800;">${c.evidence?.length || 0} AUDITED CLAIMS</span>
        </div>
        <ul class="evidence-timeline">
          ${(c.evidence || []).map(ev => `
            <li class="evidence-step">
              <p class="claim-text">${esc(ev.claim)}</p>
              <div class="provenance-tag">
                <span class="provenance-source">${esc(ev.source.toUpperCase())}</span>
                <span>·</span>
                <code>${esc(ev.ref)}</code>
                ${ev.entity_ids && ev.entity_ids.length ? `<span>·</span> <span>Entities: ${esc(ev.entity_ids.join(', '))}</span>` : ''}
              </div>
            </li>
          `).join('')}
        </ul>
      </div>
    </div>

    <!-- TAB 4: SAR REGULATORY REPORT -->
    <div class="tab-pane ${state.activeTab === 'sar' ? 'active' : ''}" id="tab-sar">
      <div class="sar-document">
        <div class="sar-doc-header">
          <div>
            <div class="sar-doc-title">SUSPICIOUS ACTIVITY REPORT (SAR) · FINCEN COMPLIANCE DRAFT</div>
            <div class="sar-fincen-tag">TigerGraph Automated Graph Investigation Audit Package</div>
          </div>
          <button id="copySarBtn" class="btn-secondary" style="padding: 0.45rem 0.85rem; font-size: 0.78rem;">
            Copy SAR
          </button>
        </div>

        <div class="sar-field-row">
          <span class="sar-field-label">FILING REQUIRED:</span>
          <span class="sar-field-val ${activeCase.sar.file ? 'text-fraud' : 'text-legit'}">
            <strong>${activeCase.sar.file ? 'YES — FILING MANDATORY' : 'NO — EXCLUDED BY POLICY R3'}</strong>
          </span>
        </div>
        <div class="sar-field-row">
          <span class="sar-field-label">SUBJECT IDENTIFIERS:</span>
          <span class="sar-field-val">${esc((activeCase.sar.subjects || [activeCase.trigger.customer_id]).join(', '))}</span>
        </div>
        <div class="sar-field-row">
          <span class="sar-field-label">AGGREGATE AMOUNT:</span>
          <span class="sar-field-val">${money(activeCase.sar.total_amount_usd || c.exposure_usd)}</span>
        </div>
        <div class="sar-field-row">
          <span class="sar-field-label">POLICY JUSTIFICATION:</span>
          <span class="sar-field-val">${esc(activeCase.sar.reason)}</span>
        </div>

        <div class="sar-narrative-block">
          <strong style="color: var(--amber-400); display: block; margin-bottom: 0.5rem; font-size: 0.9rem;">OFFICIAL SAR NARRATIVE:</strong>
          ${esc(activeCase.sar.narrative || activeCase.case.summary || 'No suspicious activity identified. Customer activity conforms to historical behavioral baseline.')}
        </div>
      </div>
    </div>
  `;

  // Attach tab switching listeners
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      state.activeTab = btn.dataset.tab;
      renderDetail();
    });
  });

  // Copy SAR listener
  const copySarBtn = $('#copySarBtn');
  if (copySarBtn) {
    copySarBtn.addEventListener('click', () => {
      const textToCopy = `CASE ${activeCase.case_id} SAR REPORT\nFiling: ${activeCase.sar.file}\nReason: ${activeCase.sar.reason}\nNarrative: ${activeCase.sar.narrative || activeCase.case.summary}`;
      navigator.clipboard.writeText(textToCopy).then(() => {
        copySarBtn.textContent = 'Copied!';
        setTimeout(() => { copySarBtn.textContent = 'Copy SAR'; }, 2000);
      });
    });
  }
}

// Initialize KPI Statistics
function updateKPIs() {
  const total = state.cases.length;
  const fraudCases = state.cases.filter(c => c.case.verdict === 'fraud');
  const legitCases = state.cases.filter(c => c.case.verdict === 'legitimate');
  const totalExposure = state.cases.reduce((sum, c) => sum + (c.case.exposure_usd || 0), 0);

  $('#kpiTotalCount').textContent = total;
  $('#kpiFraudCount').textContent = fraudCases.length;
  $('#kpiLegitCount').textContent = legitCases.length;
  $('#kpiExposure').textContent = money(totalExposure);
}

// Setup Event Listeners
function setupEvents() {
  // Search input
  const searchInput = $('#caseSearchInput');
  const clearBtn = $('#clearSearchBtn');

  searchInput.addEventListener('input', (e) => {
    state.searchQuery = e.target.value;
    clearBtn.style.display = state.searchQuery ? 'block' : 'none';
    renderQueue();
  });

  clearBtn.addEventListener('click', () => {
    searchInput.value = '';
    state.searchQuery = '';
    clearBtn.style.display = 'none';
    renderQueue();
  });

  // Filter pills
  $$('.filter-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      $$('.filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      state.activeFilter = pill.dataset.filter;
      renderQueue();
    });
  });

  // Theme toggle
  const themeToggle = $('#themeToggle');
  themeToggle.addEventListener('click', () => {
    const isLight = document.documentElement.classList.toggle('light');
    state.theme = isLight ? 'light' : 'dark';
    localStorage.setItem('nw_theme', state.theme);
  });

  // Apply initial theme
  if (state.theme === 'light') {
    document.documentElement.classList.add('light');
  }

  // Export Bundle Button
  const exportBtn = $('#exportAllBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.cases, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `nexuswatch-investigation-bundle-${new Date().toISOString().slice(0,10)}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    });
  }
}

// Application Ingestion
async function initApp() {
  try {
    const response = await fetch('cases.json');
    if (!response.ok) throw new Error('Network response was not ok');
    state.cases = await response.json();

    const urlParams = new URLSearchParams(window.location.search);
    state.selectedId = urlParams.get('case') || state.cases[0]?.case_id || 'HHG-001';

    updateKPIs();
    setupEvents();
    renderQueue();
    renderDetail();
  } catch (err) {
    console.error('Failed to load investigation dataset:', err);
    $('#caseDetailPanel').innerHTML = `
      <div class="empty-state" style="padding: 4rem; text-align: center; color: var(--rose-400);">
        <h3>Failed to load TigerGraph investigation records</h3>
        <p style="color: var(--text-muted); margin-top: 0.5rem;">Ensure web server is running and cases.json is available.</p>
      </div>
    `;
  }
}

// Start
document.addEventListener('DOMContentLoaded', initApp);
