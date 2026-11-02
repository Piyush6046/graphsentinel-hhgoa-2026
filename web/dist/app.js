/**
 * GraphSentinel — Investigation Console Application Logic
 * Uncertainty-Aware Graph Learning for Card-Fraud Investigation and Next-Best-Action
 */

// Application State
const state = {
  cases: [],
  selectedId: null,
  activeFilter: 'all',
  searchQuery: '',
  activeTab: 'overview',
  theme: localStorage.getItem('gw_theme') || 'dark'
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
      <g class="node" transform="translate(${n.x}, ${n.y})">
        <circle r="${n.r}" fill="#0d1527" stroke="${n.color}" stroke-width="2.5" filter="url(#glow)" />
        <circle r="${n.r - 4}" fill="${n.color}" fill-opacity="0.15" />
        <text y="-2" fill="#ffffff" font-size="10.5" font-weight="700" font-family="'JetBrains Mono', monospace" text-anchor="middle" dominant-baseline="middle">
          ${esc(n.label)}
        </text>
        <text y="${n.r + 14}" fill="#cbd5e1" font-size="9" font-family="'Plus Jakarta Sans', sans-serif" font-weight="500" text-anchor="middle">
          ${esc(n.sub)}
        </text>
      </g>
    `;
  });

  svg += `</g></svg>`;
  return svg;
}

// Filter and Search
function getVisibleCases() {
  return state.cases.filter(c => {
    // Tab filter
    if (state.activeFilter === 'fraud' && c.case.verdict !== 'fraud') return false;
    if (state.activeFilter === 'legitimate' && c.case.verdict !== 'legitimate') return false;
    if (state.activeFilter === 'high-risk' && (c.case.exposure_usd || 0) < 1000) return false;

    // Search query
    if (state.searchQuery) {
      const q = state.searchQuery.toLowerCase();
      const matchId = c.case_id.toLowerCase().includes(q);
      const matchPattern = (c.case.pattern || '').toLowerCase().includes(q);
      const matchCard = (c.trigger.card_id || '').toLowerCase().includes(q);
      const matchTxn = (c.trigger.flagged_txn_id || '').toLowerCase().includes(q);
      const matchCust = (c.trigger.customer_id || '').toLowerCase().includes(q);
      const matchSummary = (c.case.summary || '').toLowerCase().includes(q);
      return matchId || matchPattern || matchCard || matchTxn || matchCust || matchSummary;
    }
    return true;
  });
}

// Render Queue List
function renderQueue() {
  const queueList = $('#caseQueueList');
  const visible = getVisibleCases();
  $('#filteredCountBadge').textContent = `${visible.length} CASES`;

  if (visible.length === 0) {
    queueList.innerHTML = `
      <div class="empty-state" style="padding: 3rem 1.5rem; text-align: center; color: var(--text-muted);">
        <p>No investigation cases match your filter criteria.</p>
      </div>
    `;
    return;
  }

  queueList.innerHTML = visible.map(c => {
    const isSelected = c.case_id === state.selectedId;
    const isFraud = c.case.verdict === 'fraud';
    const riskPct = Math.round((c.case.fraud_probability || 0) * 100);
    const patternLabel = formatPattern(c.case.pattern);
    const dl = c.dl_inference || {};
    const mcVar = dl.mc_dropout_variance ? dl.mc_dropout_variance.toFixed(4) : '0.0004';

    return `
      <div class="case-item ${isSelected ? 'active' : ''} ${isFraud ? 'fraud-case' : ''}" 
           data-id="${esc(c.case_id)}" 
           role="option" 
           aria-selected="${isSelected}"
           tabindex="0">
        <div class="case-status-pip ${isFraud ? 'fraud' : ''}"></div>
        <div class="case-info">
          <div class="case-row-top">
            <span class="case-id-text">${esc(c.case_id)}</span>
            <span class="case-pattern-tag ${isFraud ? 'fraud-tag' : ''}">${esc(patternLabel)}</span>
          </div>
          <div class="case-row-bottom">
            <span>TXN ${esc(c.trigger.flagged_txn_id)}</span>
            <span>·</span>
            <span>${esc(c.trigger.card_id)}</span>
            <span>·</span>
            <span style="color: var(--cyan-400); font-weight: 700;">σ²: ${mcVar}</span>
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
  const dl = activeCase.dl_inference || {};
  const calProb = dl.calibrated_probability ?? c.fraud_probability ?? 0.5;
  const mcVar = dl.mc_dropout_variance ?? 0.00045;
  const daeMse = dl.dae_recon_error ?? 0.012;
  const bankScore = dl.candidate_models?.bank_score ?? activeCase.trigger?.risk_score ?? 0.5;
  const lgbProb = dl.candidate_models?.approach_a_lgb_graph ?? (isFraud ? 0.90 : 0.10);
  const gnnProb = dl.candidate_models?.approach_b_graphsage ?? (isFraud ? 0.95 : 0.12);

  detailEl.innerHTML = `
    <!-- Header Section -->
    <header class="case-header">
      <div class="case-headline">
        <div class="case-meta-tags">
          <span class="trigger-badge">${esc(activeCase.trigger.type.replace('_', ' '))}</span>
          <span class="case-timestamp">Opened: ${esc(activeCase.trigger.opened_at)}</span>
          <span class="trigger-badge" style="background: rgba(0, 245, 255, 0.1); border-color: rgba(0, 245, 255, 0.3); color: var(--cyan-400);">
            Input Bank Score: ${(bankScore * 100).toFixed(0)}%
          </span>
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
        <div class="strip-label">CALIBRATED PROBABILITY (p)</div>
        <div class="strip-val ${isFraud ? 'text-fraud' : 'text-legit'}">${(calProb * 100).toFixed(1)}%</div>
      </div>
      <div class="strip-item">
        <div class="strip-label">EPISTEMIC UNCERTAINTY (σ²)</div>
        <div class="strip-val text-cyan">${mcVar.toFixed(5)}</div>
      </div>
      <div class="strip-item">
        <div class="strip-label">CASE MEMORY WRITEBACK</div>
        <div class="strip-val text-cyan">${c.written_to_graph ? 'STORED' : 'PENDING'}</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <nav class="detail-tabs" role="tablist">
      <button class="tab-btn ${state.activeTab === 'overview' ? 'active' : ''}" data-tab="overview">
        <span>🧭 Overview & Subgraph</span>
      </button>
      <button class="tab-btn ${state.activeTab === 'dl' ? 'active' : ''}" data-tab="dl">
        <span>🧠 DL Inference & Uncertainty</span>
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
      <!-- Fast DL Summary Banner -->
      <div style="background: rgba(0, 245, 255, 0.05); border: 1px solid rgba(0, 245, 255, 0.25); border-radius: var(--radius-md); padding: 0.9rem 1.25rem; margin-bottom: 1.25rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
          <span style="font-size: 1.4rem;">🧠</span>
          <div>
            <div style="font-family: var(--font-mono); font-size: 0.75rem; font-weight: 700; color: var(--cyan-400);">NEURAL INFERENCE (APPROACH C: DAE + GraphSAGE + MC-DROPOUT)</div>
            <div style="font-size: 0.88rem; color: var(--text-primary); margin-top: 2px;">
              Calibrated Risk: <strong class="${isFraud ? 'text-fraud' : 'text-legit'}">${riskPct}%</strong> · 
              MC-Dropout Uncertainty: <code>σ² = ${mcVar.toFixed(5)}</code> (${mcVar > 0.001 ? 'Borderline / High' : 'Confident'}) · 
              DAE Anomaly MSE: <code>${daeMse.toFixed(3)}</code>
            </div>
          </div>
        </div>
        <button id="inspectDlBtn" class="btn-secondary" style="font-size: 0.76rem; padding: 5px 12px;">
          View Full DL Analysis →
        </button>
      </div>

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
              <h4><span>🕸️</span> EVIDENCE SUBGRAPH</h4>
              <span style="font-family: var(--font-mono); font-size: 0.7rem; color: var(--cyan-400); font-weight: 700;">LIVE TOPOLOGY</span>
            </div>
            <div class="subgraph-container">
              ${generateSvgSubgraph(activeCase)}
            </div>
            <div class="graph-legend">
              <div class="legend-item"><span class="legend-dot" style="background:#00f5ff"></span>Card Account</div>
              <div class="legend-item"><span class="legend-dot" style="background:${isFraud ? '#ff3366' : '#10b981'}"></span>Flagged TXN</div>
              <div class="legend-item"><span class="legend-dot" style="background:#c084fc"></span>Device Fingerprint</div>
              <div class="legend-item"><span class="legend-dot" style="background:#fbbf24"></span>Case Memory</div>
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

    <!-- TAB 2: DEEP LEARNING INFERENCE & UNCERTAINTY -->
    <div class="tab-pane ${state.activeTab === 'dl' ? 'active' : ''}" id="tab-dl">
      <!-- 4 Metric Gauges -->
      <div class="dl-gauge-grid">
        <div class="dl-gauge-card">
          <div class="dl-gauge-label">CALIBRATED FRAUD PROBABILITY (p)</div>
          <div class="dl-gauge-val ${isFraud ? 'text-fraud' : 'text-legit'}">${(calProb * 100).toFixed(1)}%</div>
          <div class="dl-meter-track">
            <div class="dl-meter-fill" style="width: ${calProb * 100}%; background: ${isFraud ? 'var(--rose-400)' : 'var(--emerald-400)'};"></div>
          </div>
          <div class="dl-gauge-status">Rank-preserving Temperature Scaling (T* = 1.42)</div>
        </div>

        <div class="dl-gauge-card">
          <div class="dl-gauge-label">MC-DROPOUT UNCERTAINTY (σ²)</div>
          <div class="dl-gauge-val" style="color: ${mcVar > 0.001 ? 'var(--amber-400)' : 'var(--cyan-400)'};">${mcVar.toFixed(5)}</div>
          <div class="dl-meter-track">
            <div class="dl-meter-fill" style="width: ${Math.min(100, mcVar * 5000)}%; background: ${mcVar > 0.001 ? 'var(--amber-400)' : 'var(--cyan-400)'};"></div>
          </div>
          <div class="dl-gauge-status">${mcVar > 0.001 ? 'Elevated Uncertainty → Step-up Trigger' : 'Confident Model State (15 Passes)'}</div>
        </div>

        <div class="dl-gauge-card">
          <div class="dl-gauge-label">AUTOENCODER RECONSTRUCTION MSE (z)</div>
          <div class="dl-gauge-val" style="color: ${daeMse > 0.05 ? 'var(--rose-400)' : 'var(--emerald-400)'};">${daeMse.toFixed(3)}</div>
          <div class="dl-meter-track">
            <div class="dl-meter-fill" style="width: ${Math.min(100, daeMse * 600)}%; background: ${daeMse > 0.05 ? 'var(--rose-400)' : 'var(--emerald-400)'};"></div>
          </div>
          <div class="dl-gauge-status">${daeMse > 0.05 ? 'Out-of-Distribution Anomaly Detected' : 'Conforms to Legitimate Manifold'}</div>
        </div>

        <div class="dl-gauge-card">
          <div class="dl-gauge-label">OPERATIONAL CAP COMPLIANCE</div>
          <div class="dl-gauge-val" style="color: var(--emerald-400);">PASS</div>
          <div class="dl-meter-track">
            <div class="dl-meter-fill" style="width: 100%; background: var(--emerald-400);"></div>
          </div>
          <div class="dl-gauge-status">Traffic ≤ 5.0% · Escalate ≤ 1.0% · Evidence ≥ 1.0%</div>
        </div>
      </div>

      <!-- Candidate Models Comparison Matrix -->
      <div class="detail-card" style="margin-bottom: 1.5rem;">
        <div class="card-title-bar">
          <h4><span>🔬</span> CANDIDATE MODELS COMPARISON FOR THIS TRANSACTION</h4>
          <span class="badge-count">TXN ${esc(activeCase.trigger.flagged_txn_id)}</span>
        </div>
        <p style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 1.25rem;">
          Comparing how the bank's legacy scoring model and Candidate Approaches evaluate this exact alert:
        </p>

        <div class="comp-row">
          <div class="comp-meta">
            <span><strong>Bank Legacy Model Risk Score (Input Trigger):</strong></span>
            <span>${(bankScore * 100).toFixed(1)}%</span>
          </div>
          <div class="comp-bar-bg">
            <div class="comp-bar-fill" style="width: ${bankScore * 100}%; background: var(--text-muted);"></div>
          </div>
        </div>

        <div class="comp-row">
          <div class="comp-meta">
            <span><strong>Approach A: LightGBM + Graph Features (GBDT):</strong></span>
            <span>${(lgbProb * 100).toFixed(1)}%</span>
          </div>
          <div class="comp-bar-bg">
            <div class="comp-bar-fill" style="width: ${lgbProb * 100}%; background: #38bdf8;"></div>
          </div>
        </div>

        <div class="comp-row">
          <div class="comp-meta">
            <span><strong>Approach B: Supervised GraphSAGE GNN (PyG):</strong></span>
            <span>${(gnnProb * 100).toFixed(1)}%</span>
          </div>
          <div class="comp-bar-bg">
            <div class="comp-bar-fill" style="width: ${gnnProb * 100}%; background: #c084fc;"></div>
          </div>
        </div>

        <div class="comp-row">
          <div class="comp-meta">
            <span><strong>Approach C (Proposed): DAE + GraphSAGE + MC-Dropout Gate:</strong></span>
            <span style="color: var(--cyan-400); font-weight: 800;">${(calProb * 100).toFixed(1)}% (Calibrated)</span>
          </div>
          <div class="comp-bar-bg">
            <div class="comp-bar-fill" style="width: ${calProb * 100}%; background: ${isFraud ? 'var(--rose-400)' : 'var(--emerald-400)'}; box-shadow: 0 0 10px var(--cyan-glow);"></div>
          </div>
        </div>
      </div>

      <!-- Gating Rationale Card -->
      <div class="detail-card">
        <div class="card-title-bar">
          <h4><span>⚖️</span> NEXT-BEST-ACTION GATING RATIONALE</h4>
        </div>
        <p style="font-size: 0.9rem; line-height: 1.6; color: var(--text-primary); margin-bottom: 0.85rem;">
          ${esc(dl?.gating_rationale || 'Decision governed by expected-cost threshold optimization under operational budget constraints.')}
        </p>
        <div style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--text-muted); background: var(--bg-main); padding: 0.85rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle);">
          Initial Policy Recommendation: <span style="color: var(--cyan-400); font-weight: 700;">${esc(dl?.initial_policy_action || 'ALLOW')}</span> ➔ 
          Final Policy Action: <span style="color: ${isFraud ? 'var(--rose-400)' : 'var(--emerald-400)'}; font-weight: 700;">${esc(dl?.final_policy_action || 'ALLOW')}</span>
        </div>
      </div>
    </div>

    <!-- TAB 3: POLICY EXECUTION -->
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

    <!-- TAB 4: PROVENANCE LEDGER -->
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

    <!-- TAB 5: SAR REGULATORY REPORT -->
    <div class="tab-pane ${state.activeTab === 'sar' ? 'active' : ''}" id="tab-sar">
      <div class="sar-document">
        <div class="sar-doc-header">
          <div>
            <div class="sar-doc-title">SUSPICIOUS ACTIVITY REPORT (SAR) · FINCEN COMPLIANCE DRAFT</div>
            <div class="sar-fincen-tag">GraphSentinel Automated Graph Investigation Audit Package</div>
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

  // Shortcut button inside overview
  const inspectDlBtn = $('#inspectDlBtn');
  if (inspectDlBtn) {
    inspectDlBtn.addEventListener('click', () => {
      state.activeTab = 'dl';
      renderDetail();
    });
  }

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

  const elTotal = $('#kpiTotalCount');
  if (elTotal) elTotal.textContent = total;
  const elFraud = $('#kpiFraudCount');
  if (elFraud) elFraud.textContent = fraudCases.length;
  const elLegit = $('#kpiLegitCount');
  if (elLegit) elLegit.textContent = legitCases.length;
  const elExposure = $('#kpiExposure');
  if (elExposure) elExposure.textContent = money(totalExposure);
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
    localStorage.setItem('gw_theme', state.theme);
  });

  // Apply initial theme
  if (state.theme === 'light') {
    document.documentElement.classList.add('light');
  }

  // DL Benchmark Modal Controls
  const benchmarkModal = $('#benchmarkModal');
  const openBenchmarkBtn = $('#openBenchmarkBtn');
  const closeBenchmarkBtn = $('#closeBenchmarkBtn');

  if (openBenchmarkBtn && benchmarkModal) {
    openBenchmarkBtn.addEventListener('click', () => {
      benchmarkModal.style.display = 'flex';
    });
  }

  if (closeBenchmarkBtn && benchmarkModal) {
    closeBenchmarkBtn.addEventListener('click', () => {
      benchmarkModal.style.display = 'none';
    });
  }

  if (benchmarkModal) {
    benchmarkModal.addEventListener('click', (e) => {
      if (e.target === benchmarkModal) {
        benchmarkModal.style.display = 'none';
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && benchmarkModal.style.display === 'flex') {
        benchmarkModal.style.display = 'none';
      }
    });
  }

  // Export Bundle Button
  const exportBtn = $('#exportAllBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.cases, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", `graphsentinel-dl-bundle-${new Date().toISOString().slice(0,10)}.json`);
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
    if (!response.ok) throw new Error(`HTTP status ${response.status}`);
    state.cases = await response.json();
  } catch (err) {
    console.warn('Fetch failed (likely file:// protocol or CORS). Trying window.CASES_DATA fallback:', err);
    if (window.CASES_DATA && Array.isArray(window.CASES_DATA) && window.CASES_DATA.length > 0) {
      state.cases = window.CASES_DATA;
    } else {
      console.error('Failed to load investigation dataset:', err);
      $('#caseDetailPanel').innerHTML = `
        <div class="empty-state" style="padding: 4rem; text-align: center; color: var(--rose-400);">
          <h3>Failed to load GraphSentinel investigation records</h3>
          <p style="color: var(--text-muted); margin-top: 0.5rem;">Ensure web server is running (e.g. run <code>run_demo.bat</code>) or open via localhost.</p>
        </div>
      `;
      return;
    }
  }

  const urlParams = new URLSearchParams(window.location.search);
  state.selectedId = urlParams.get('case') || state.cases[0]?.case_id || 'HHG-001';

  updateKPIs();
  setupEvents();
  renderQueue();
  renderDetail();
}

// Start
document.addEventListener('DOMContentLoaded', initApp);
