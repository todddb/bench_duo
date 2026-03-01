const api = async (path, opts = {}) => {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const text = await res.text();

  let payload;
  try {
    payload = JSON.parse(text);
  } catch (err) {
    console.error('Non-JSON response:', text);
    throw new Error('Server returned invalid response. Check backend logs.');
  }

  if (!res.ok || payload.success === false) {
    throw new Error(payload.error || `Request failed: ${res.status}`);
  }

  if (Object.prototype.hasOwnProperty.call(payload, 'data')) {
    return payload.data;
  }
  return payload;
};

const ENGINE_DEFAULT_PORTS = {
  ollama: 11434,
  mlx: 8000,
  'tensorrt-llm': 8001,
};

const statusDot = (status) => `<span class="status-dot status-${status || 'red'}"></span>`;
const byId = (id) => document.getElementById(id);


const warmBadge = (status) => {
  if (status === 'warm') return '<span class="badge text-bg-success">🟢 Warm</span>';
  if (status === 'loading') return '<span class="badge text-bg-warning">🟡 Loading</span>';
  if (status === 'error') return '<span class="badge text-bg-danger">🔴 Error</span>';
  return '<span class="badge text-bg-secondary">🔴 Cold</span>';
};

const isCompatibleAgent = (agent, activeEngine) => {
  if (!activeEngine) return true;
  return (agent.engine || '').toLowerCase() === activeEngine.toLowerCase();
};

const engineTooltipText = (engine) => {
  if (engine.reachable) return `Inference engine reachable. Last checked ${engine.last_checked || 'never'}.`;
  return `Inference engine unreachable at ${engine.host}. Last checked ${engine.last_checked || 'never'}. Click for diagnostics and retry.`;
};

const modelTooltipText = (model) => {
  if (model.load_state === 'not_present') return 'Model files not found on host. Add model files or update model path.';
  if (model.load_state === 'cold') return 'Model present on disk but not loaded in engine. Click to load.';
  return `Model loaded in inference engine (cached). Last loaded ${model.last_load_attempt || 'unknown'}.`;
};

const agentTooltipText = (status) => {
  if (status === 'ready') return 'Agent ready to accept queries: engine reachable and model loaded.';
  if (status === 'partially_ready') return 'Agent is configured but runtime unavailable (engine unreachable or model not loaded). Click for diagnostics.';
  if (status === 'not_ready') return 'Agent cannot run (model missing or configuration error). Click to view error.';
  return 'Agent is disabled.';
};

const agentPillClass = (status) => {
  if (status === 'ready') return 'text-bg-success';
  if (status === 'partially_ready') return 'text-bg-warning';
  if (status === 'not_ready') return 'text-bg-danger';
  return 'text-bg-secondary';
};

let activeStatusPopover = null;
let popoverTimer = null;

const closeStatusPopover = () => {
  if (activeStatusPopover) {
    activeStatusPopover.remove();
    activeStatusPopover = null;
  }
  if (popoverTimer) {
    clearTimeout(popoverTimer);
    popoverTimer = null;
  }
};

document.addEventListener('click', (event) => {
  if (activeStatusPopover && !activeStatusPopover.contains(event.target) && !event.target.closest('.status-trigger')) {
    closeStatusPopover();
  }
});

const renderStatusPopover = ({ anchorEl, status, modelId, onStatusChanged }) => {
  closeStatusPopover();
  const rect = anchorEl.getBoundingClientRect();
  const pop = document.createElement('div');
  pop.className = 'status-popover card shadow-sm p-2';
  pop.style.left = `${rect.left + window.scrollX}px`;
  pop.style.top = `${rect.bottom + window.scrollY + 8}px`;

  const recentLogs = (status.logs?.recent || []).slice(-5).map((line) => `<li class="small"><code>${line}</code></li>`).join('') || '<li class="small text-muted">No recent logs</li>';
  pop.innerHTML = `
    <div class="small"><strong>Engine:</strong> ${status.engine.message || (status.engine.reachable ? 'ok' : 'unreachable')}<br/><span class="text-muted">${status.engine.last_checked || 'never'}</span></div>
    <div class="small mt-2"><strong>Model load:</strong> ${status.model.last_load_message || status.model.load_state}<br/><span class="text-muted">${status.model.last_load_attempt || 'never'}</span></div>
    <div class="small mt-2"><strong>Recent log snippet</strong><ul class="mb-2 ps-3">${recentLogs}</ul></div>
    <div class="d-flex gap-2 flex-wrap">
      <button class="btn btn-sm btn-outline-secondary" data-action="engine">Retry engine check</button>
      <button class="btn btn-sm btn-outline-primary" data-action="reload" ${status.model.load_state === 'not_present' ? 'disabled' : ''}>Reload model</button>
      <button class="btn btn-sm btn-outline-dark" data-action="logs">View agent logs</button>
    </div>
  `;
  document.body.appendChild(pop);
  activeStatusPopover = pop;
  popoverTimer = setTimeout(closeStatusPopover, 30000);

  pop.querySelector('[data-action="engine"]').onclick = async () => {
    await api('/api/v1/engine/check', { method: 'POST', body: JSON.stringify({ model_id: modelId }) });
    await onStatusChanged();
    closeStatusPopover();
  };
  pop.querySelector('[data-action="reload"]').onclick = async () => {
    await api(`/api/v1/models/${modelId}/reload`, { method: 'POST' });
    await onStatusChanged();
    closeStatusPopover();
  };
  pop.querySelector('[data-action="logs"]').onclick = () => {
    alert((status.logs?.recent || []).join('\n') || 'No logs found');
  };
};

async function warmModelById(modelId) {
  return api('/api/models/warm', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId }),
  });
}

async function initSetupPage() {
  const modelModal = new bootstrap.Modal(byId('model-modal'));
  const agentModal = new bootstrap.Modal(byId('agent-modal'));

  const loadModels = async () => {
    const models = await api('/api/models');
    const statuses = await Promise.all(models.map(async (m) => {
      try {
        const status = await api(`/api/v1/models/${m.id}/status`);
        return [m.id, status];
      } catch (err) {
        return [m.id, null];
      }
    }));
    const statusMap = Object.fromEntries(statuses);

    byId('model-list').innerHTML = models.map((m) => {
      const status = statusMap[m.id];
      const engine = status?.engine;
      const model = status?.model;
      const engineSignal = engine ? `<button class="status-trigger engine-signal ${engine.reachable ? 'is-ok' : 'is-bad'}" title="${engineTooltipText(engine)}" data-model-id="${m.id}" data-status-target="engine"></button>` : statusDot(m.status);
      const loadBadge = model ? `<button class="status-trigger btn btn-sm ${model.load_state === 'warm' ? 'btn-success' : (model.load_state === 'cold' ? 'btn-secondary' : 'btn-danger')}" title="${modelTooltipText(model)}" data-model-id="${m.id}" data-status-target="model">${model.load_state === 'warm' ? 'Warm' : (model.load_state === 'cold' ? 'Cold' : 'Not Present')}</button>` : warmBadge(m.warm_status);
      return `
      <div class="list-group-item d-flex justify-content-between align-items-center">
        <div>
          ${engineSignal}<strong>${m.name}</strong>
          <small class="text-muted">${m.host}:${m.port} • ${m.backend}/${m.model_name}</small>
          <div class="mt-1">${loadBadge}</div>
        </div>
        <div>
          <button class="btn btn-sm btn-outline-primary" onclick='window.loadModel(${m.id})'>Load Model</button>
          <button class="btn btn-sm btn-outline-secondary" onclick='window.editModel(${JSON.stringify(m)})'>Edit</button>
          <button class="btn btn-sm btn-outline-danger" onclick='window.deleteModel(${m.id})'>Delete</button>
        </div>
      </div>
    `;
    }).join('');

    document.querySelectorAll('#model-list .status-trigger').forEach((trigger) => {
      trigger.onclick = async (event) => {
        const modelId = Number(event.currentTarget.dataset.modelId);
        const status = await api(`/api/v1/models/${modelId}/status`);
        renderStatusPopover({ anchorEl: event.currentTarget, status, modelId, onStatusChanged: async () => { await loadModels(); await loadAgents(); } });
      };
    });

    const active = models.find((m) => m.status === 'green');
    const activeEngine = active ? (active.engine || active.backend) : null;
    const select = byId('agent-model-id');
    select.innerHTML = models.map((m) => {
      const modelEngine = (m.engine || m.backend || '').toLowerCase();
      const disabled = activeEngine && modelEngine !== activeEngine.toLowerCase();
      const label = `${m.name} (${m.model_name})${disabled ? ' — incompatible engine' : ''}`;
      return `<option value="${m.id}" ${disabled ? 'disabled' : ''}>${label}</option>`;
    }).join('');
  };

  const loadAgents = async () => {
    const agents = await api('/api/agents');
    const statuses = await Promise.all(agents.map(async (a) => {
      try {
        const status = await api(`/api/v1/agents/${a.id}/status`);
        return [a.id, status];
      } catch (err) {
        return [a.id, null];
      }
    }));
    const statusMap = Object.fromEntries(statuses);

    byId('agent-list').innerHTML = agents.map((a) => {
      const s = statusMap[a.id];
      const agg = s?.agent?.status || a.aggregate_status || 'partially_ready';
      return `
      <div class="list-group-item d-flex justify-content-between align-items-center">
        <div><span class="badge ${agentPillClass(agg)} status-trigger" title="${agentTooltipText(agg)}" data-agent-id="${a.id}">${agg.replace('_', ' ')}</span> <strong>${a.name}</strong> <small class="text-muted">${a.model_name || a.model_id}</small></div>
        <div>
          <button class="btn btn-sm btn-outline-secondary" onclick='window.editAgent(${JSON.stringify(a)})'>Edit</button>
          <button class="btn btn-sm btn-outline-danger" onclick='window.deleteAgent(${a.id})'>Delete</button>
        </div>
      </div>
    `;
    }).join('');

    document.querySelectorAll('#agent-list .status-trigger').forEach((trigger) => {
      trigger.onclick = async (event) => {
        const agentId = Number(event.currentTarget.dataset.agentId);
        const status = await api(`/api/v1/agents/${agentId}/status`);
        renderStatusPopover({ anchorEl: event.currentTarget, status, modelId: status?.model_id || Number(status?.model?.id || 0) || Number(agents.find((a) => a.id === agentId)?.model_id), onStatusChanged: async () => { await loadModels(); await loadAgents(); } });
      };
    });
  };

  const modelEngine = byId('model-engine');
  const modelNameSelect = byId('model-model-name');
  const saveModelBtn = byId('save-model-btn');
  const testModelBtn = byId('test-model-btn');
  const modelTestStatus = byId('model-test-status');

  const setModelTestState = (state, message) => {
    modelTestStatus.className = 'badge';
    if (state === 'success') modelTestStatus.classList.add('text-bg-success');
    else if (state === 'error') modelTestStatus.classList.add('text-bg-danger');
    else if (state === 'loading') modelTestStatus.classList.add('text-bg-info');
    else modelTestStatus.classList.add('text-bg-secondary');
    modelTestStatus.textContent = message;
  };

  const resetModelDetection = () => {
    modelNameSelect.innerHTML = '';
    modelNameSelect.disabled = true;
    saveModelBtn.disabled = true;
    testModelBtn.disabled = false;
    setModelTestState('idle', 'Not tested');
  };

  const setDetectionOptions = (models) => {
    const modelOptions = models.length ? models : ['No models found'];
    modelNameSelect.innerHTML = modelOptions.map((model) => `<option value="${model}">${model}</option>`).join('');
    modelNameSelect.disabled = !models.length;
    saveModelBtn.disabled = !models.length;
  };

  const testBackend = async () => {
    const host = byId('model-host').value;
    const rawPort = byId('model-port').value;
    const port = rawPort ? Number(rawPort) : 9001;

    setModelTestState('loading', 'Testing...');
    testModelBtn.disabled = true;

    try {
      const result = await api('/api/models/probe', {
        method: 'POST',
        body: JSON.stringify({ host, port, engine: modelEngine.value }),
      });
      const models = result.models || [];
      setDetectionOptions(models);
      setModelTestState('success', `Detected ${modelEngine.value}`);
    } catch (err) {
      resetModelDetection();
      setModelTestState('error', err.message);
      alert(err.message);
    } finally {
      testModelBtn.disabled = false;
    }
  };

  byId('add-model-btn').onclick = () => {
    byId('model-form').reset();
    byId('model-id').value = '';
    modelEngine.value = 'ollama';
    byId('model-port').value = ENGINE_DEFAULT_PORTS.ollama;
    resetModelDetection();
    modelModal.show();
  };
  byId('add-agent-btn').onclick = () => { byId('agent-form').reset(); byId('agent-id').value = ''; agentModal.show(); };
  byId('refresh-models').onclick = async () => { await loadModels(); await loadAgents(); };

  window.editModel = (m) => {
    byId('model-id').value = m.id;
    byId('model-name').value = m.name;
    byId('model-host').value = m.host;
    modelEngine.value = m.engine || m.backend || 'ollama';
    byId('model-port').value = m.port || ENGINE_DEFAULT_PORTS[modelEngine.value] || 11434;
    setDetectionOptions(m.model_name ? [m.model_name] : []);
    if (m.model_name) {
      modelNameSelect.value = m.model_name;
      saveModelBtn.disabled = false;
      setModelTestState('success', `Loaded ${modelEngine.value}`);
    } else {
      setModelTestState('idle', 'Not tested');
    }
    modelModal.show();
  };
  window.deleteModel = async (id) => { if (confirm('Delete model?')) { await api(`/api/models/${id}`, { method: 'DELETE' }); await loadModels(); await loadAgents(); } };
  window.loadModel = async (modelId) => {
    await warmModelById(modelId);
    await loadModels();
    await loadAgents();
  };

  testModelBtn.onclick = testBackend;

  modelEngine.addEventListener('change', async function onEngineChange() {
    const engine = this.value;
    const defaultPort = ENGINE_DEFAULT_PORTS[engine];
    if (defaultPort) byId('model-port').value = defaultPort;
    resetModelDetection();
    await testBackend();
  });


  window.editAgent = (a) => {
    byId('agent-id').value = a.id; byId('agent-name').value = a.name; byId('agent-model-id').value = a.model_id;
    byId('agent-system-prompt').value = a.system_prompt; byId('agent-max-tokens').value = a.max_tokens; byId('agent-temperature').value = a.temperature; agentModal.show();
  };
  window.deleteAgent = async (id) => { if (confirm('Delete agent?')) { await api(`/api/agents/${id}`, { method: 'DELETE' }); await loadAgents(); } };

  byId('model-form').onsubmit = async (e) => {
    e.preventDefault();
    const id = byId('model-id').value;
    const body = {
      name: byId('model-name').value,
      host: byId('model-host').value,
      port: Number(byId('model-port').value || 9001),
      engine: modelEngine.value,
      model_name: byId('model-model-name').value,
      selected_model: byId('model-model-name').value,
    };
    await api(id ? `/api/models/${id}` : '/api/models', { method: id ? 'PUT' : 'POST', body: JSON.stringify(body) });
    modelModal.hide(); await loadModels(); await loadAgents();
  };

  byId('agent-form').onsubmit = async (e) => {
    e.preventDefault();
    const id = byId('agent-id').value;
    const body = {
      name: byId('agent-name').value, model_id: Number(byId('agent-model-id').value), system_prompt: byId('agent-system-prompt').value,
      max_tokens: Number(byId('agent-max-tokens').value), temperature: Number(byId('agent-temperature').value),
    };
    await api(id ? `/api/agents/${id}` : '/api/agents', { method: id ? 'PUT' : 'POST', body: JSON.stringify(body) });
    agentModal.hide(); await loadAgents();
  };

  await loadModels();
  await loadAgents();
}

async function initChatPage() {
  const pane = byId('chat-pane');
  const startBtn = byId('start-chat');
  const warmState = byId('chat-warm-state');

  let modelById = {};
  let activeEngine = null;

  const setWarmState = (text, loading = false) => {
    warmState.textContent = text;
    warmState.className = loading ? 'small text-warning' : 'small text-muted';
  };

  const loadAgentsForChat = async () => {
    const [agents, models] = await Promise.all([api('/api/agents'), api('/api/models')]);
    modelById = Object.fromEntries(models.map((m) => [m.id, m]));
    const active = models.find((m) => m.status === 'green');
    activeEngine = active ? (active.engine || active.backend || '').toLowerCase() : null;

    const statusEntries = await Promise.all(agents.map(async (a) => {
      try {
        const status = await api(`/api/v1/agents/${a.id}/status`);
        return [a.id, status.agent.status];
      } catch (err) {
        return [a.id, 'partially_ready'];
      }
    }));
    const statusMap = Object.fromEntries(statusEntries);

    const options = agents.map((a) => {
      const engine = (a.engine || '').toLowerCase();
      const status = statusMap[a.id] || 'partially_ready';
      const mismatch = activeEngine && engine !== activeEngine;
      const blocked = mismatch || status === 'partially_ready' || status === 'not_ready' || status === 'disabled';
      return `<option value="${a.id}" data-model-id="${a.model_id}" ${blocked ? 'disabled' : ''}>${a.name}${blocked ? ` (${status.replace('_', ' ')}, retry required)` : ''}</option>`;
    }).join('');
    byId('chat-agent1').innerHTML = options;
    byId('chat-agent2').innerHTML = options;

    const list = byId('chat-agent-status-list');
    if (list) {
      list.innerHTML = agents.map((a) => {
        const status = statusMap[a.id] || 'partially_ready';
        return `<li class="list-group-item d-flex justify-content-between"><span>${a.name}</span><span class="badge ${agentPillClass(status)}">${status.replace('_', ' ')}</span></li>`;
      }).join('');
    }
  };

  const ensureChatWarm = async () => {
    const selected = [byId('chat-agent1'), byId('chat-agent2')];
    startBtn.disabled = true;
    for (const select of selected) {
      const modelId = Number(select.selectedOptions[0]?.dataset.modelId || 0);
      const model = modelById[modelId];
      if (!model) continue;
      if (model.warm_status !== 'warm') {
        setWarmState(`Warming ${model.name}...`, true);
        await warmModelById(modelId);
      }
    }
    await loadAgentsForChat();
    setWarmState('Ready');
    startBtn.disabled = false;
  };

  await loadAgentsForChat();
  setWarmState('Ready');

  const socket = io('/');
  let typingEl = null;
  let currentConversationId = null;
  const scrollDown = () => { pane.scrollTop = pane.scrollHeight; };
  const bubble = (sender, text) => `<div class="bubble ${sender}">${text}</div>`;

  socket.on('chat_message', (msg) => {
    if (typingEl) { typingEl.remove(); typingEl = null; }
    pane.insertAdjacentHTML('beforeend', bubble(msg.sender, msg.text));
    if (!msg.done) {
      typingEl = document.createElement('div');
      typingEl.className = 'typing';
      typingEl.innerHTML = '<span></span><span></span><span></span>';
      pane.appendChild(typingEl);
    }
    scrollDown();
  });
  socket.on('chat_end', () => { if (typingEl) { typingEl.remove(); typingEl = null; } });

  byId('chat-agent1').onchange = ensureChatWarm;
  byId('chat-agent2').onchange = ensureChatWarm;

  byId('start-chat').onclick = async () => {
    await ensureChatWarm();
    pane.innerHTML = '';
    currentConversationId = null;
    byId('export-conversation-json').disabled = true;
    byId('export-conversation-csv').disabled = true;
    pane.insertAdjacentHTML('beforeend', bubble('user', byId('chat-prompt').value || 'Start.'));
    typingEl = document.createElement('div');
    typingEl.className = 'typing';
    typingEl.innerHTML = '<span></span><span></span><span></span>';
    pane.appendChild(typingEl);
    socket.emit('start_chat', {
      agent1_id: Number(byId('chat-agent1').value),
      agent2_id: Number(byId('chat-agent2').value),
      prompt: byId('chat-prompt').value,
      ttl: Number(byId('chat-ttl').value),
      seed: Date.now(),
    }, (ack) => {
      if (ack?.conversation_id) {
        currentConversationId = ack.conversation_id;
        byId('export-conversation-json').disabled = false;
        byId('export-conversation-csv').disabled = false;
      }
    });
    scrollDown();
  };

  byId('export-conversation-json').onclick = () => {
    if (currentConversationId) window.open(`/api/conversations/${currentConversationId}/export?format=json`, '_blank');
  };
  byId('export-conversation-csv').onclick = () => {
    if (currentConversationId) window.open(`/api/conversations/${currentConversationId}/export?format=csv`, '_blank');
  };
}

async function initBatchPage() {
  const startBatchBtn = byId('start-batch');
  const warmState = byId('batch-warm-state');

  let activeBatchId = null;
  let modelById = {};
  let activeEngine = null;

  const setWarmState = (text, loading = false) => {
    warmState.textContent = text;
    warmState.className = loading ? 'small text-warning mb-2' : 'small text-muted mb-2';
  };

  const loadAgentsForBatch = async () => {
    const [agents, models] = await Promise.all([api('/api/agents'), api('/api/models')]);
    modelById = Object.fromEntries(models.map((m) => [m.id, m]));
    const active = models.find((m) => m.status === 'green');
    activeEngine = active ? (active.engine || active.backend || '').toLowerCase() : null;
    const alive = agents.filter((a) => a.effective_status === 'green');
    const options = alive.map((a) => {
      const engine = (a.engine || '').toLowerCase();
      const disabled = activeEngine && engine !== activeEngine;
      return `<option value="${a.id}" data-model-id="${a.model_id}" ${disabled ? 'disabled' : ''}>${a.name}${disabled ? ' (engine mismatch)' : ''}</option>`;
    }).join('');
    byId('batch-agent1').innerHTML = options;
    byId('batch-agent2').innerHTML = options;
  };

  const ensureBatchWarm = async () => {
    startBatchBtn.disabled = true;
    for (const select of [byId('batch-agent1'), byId('batch-agent2')]) {
      const modelId = Number(select.selectedOptions[0]?.dataset.modelId || 0);
      const model = modelById[modelId];
      if (!model) continue;
      if (model.warm_status !== 'warm') {
        setWarmState(`Warming ${model.name}...`, true);
        await warmModelById(modelId);
      }
    }
    await loadAgentsForBatch();
    setWarmState('Ready');
    startBatchBtn.disabled = false;
  };

  await loadAgentsForBatch();
  setWarmState('Ready');

  const renderBatch = (detail) => {
    const s = detail.summary || {};
    const pct = s.progress_pct || 0;
    byId('export-batch-json').disabled = false;
    byId('export-batch-csv').disabled = false;
    byId('batch-progress').style.width = `${pct}%`;
    byId('batch-progress').textContent = `${pct}%`;
    byId('batch-stats').textContent = `Completed ${s.completed || 0}/${s.total || detail.num_runs}; avg time: ${s.avg_time || 0}s; speed: ${s.tokens_per_sec || 0} tok/s`;
  };

  const refreshActiveBatch = async () => {
    if (!activeBatchId) return;
    const detail = await api(`/api/batch_jobs/${activeBatchId}`);
    renderBatch(detail);
    if (!['queued', 'running'].includes(detail.status)) {
      activeBatchId = null;
      clearInterval(pollTimer);
    }
  };

  const renderHistory = async () => {
    const jobs = await api('/api/batch_jobs');
    byId('batch-table').querySelector('tbody').innerHTML = jobs.map((j) => {
      const avg = j.summary?.tokens_per_sec ?? 0;
      const date = j.created_at ? new Date(j.created_at).toLocaleString() : '-';
      return `<tr data-id="${j.id}" class="batch-row"><td>${j.id}</td><td>${j.agent1_name}/${j.agent2_name}</td><td>${date}</td><td>${j.prompt_snippet}</td><td>${j.num_runs}</td><td>${j.elapsed_seconds || 0}</td><td>${avg}</td><td>${j.status}</td></tr>`;
    }).join('');
    document.querySelectorAll('.batch-row').forEach((row) => {
      row.onclick = async () => {
        activeBatchId = Number(row.dataset.id);
        const detail = await api(`/api/batch_jobs/${activeBatchId}`);
        renderBatch(detail);
      };
    });
  };

  byId('batch-agent1').onchange = ensureBatchWarm;
  byId('batch-agent2').onchange = ensureBatchWarm;

  byId('start-batch').onclick = async () => {
    await ensureBatchWarm();
    byId('export-batch-json').disabled = true;
    byId('export-batch-csv').disabled = true;
    const created = await api('/api/batch_jobs', {
      method: 'POST',
      body: JSON.stringify({
        agent1_id: Number(byId('batch-agent1').value),
        agent2_id: Number(byId('batch-agent2').value),
        ttl: Number(byId('batch-ttl').value),
        num_runs: Number(byId('batch-runs').value),
        prompt: byId('batch-prompt').value,
        seed: Date.now(),
      }),
    });
    activeBatchId = created.batch_id;
    await renderHistory();
    await refreshActiveBatch();
  };

  byId('stop-batch').onclick = async () => {
    if (activeBatchId) {
      await api(`/api/batch_jobs/${activeBatchId}/cancel`, { method: 'POST' });
      await refreshActiveBatch();
      await renderHistory();
    }
  };


  byId('export-batch-json').onclick = () => {
    if (activeBatchId) window.open(`/api/batch_jobs/${activeBatchId}/export?format=json`, '_blank');
  };
  byId('export-batch-csv').onclick = () => {
    if (activeBatchId) window.open(`/api/batch_jobs/${activeBatchId}/export?format=csv`, '_blank');
  };

  const pollTimer = setInterval(async () => {
    await renderHistory();
    await refreshActiveBatch();
  }, 1500);

  await renderHistory();
}

async function initEvaluationPage() {
  const [models, conversations] = await Promise.all([api('/api/models'), api('/api/conversations')]);
  byId('eval-main-model').innerHTML = models.map((m) => `<option value="${m.id}">${m.name}</option>`).join('');
  byId('eval-judges').innerHTML = models.map((m) => `<option value="${m.id}">${m.name}</option>`).join('');
  byId('eval-conversation').innerHTML = conversations.map((c) => `<option value="${c.id}">Conversation #${c.id} (${c.status})</option>`).join('');

  const loadConversation = async (conversationId) => {
    const messages = await api(`/api/conversations/${conversationId}/messages`);
    byId('eval-chat').innerHTML = messages.map((m) => `<div id="msg-${m.id}" class="bubble ${m.sender_role}">${m.content}</div>`).join('');
  };

  byId('eval-conversation').onchange = async () => loadConversation(byId('eval-conversation').value);
  if (conversations.length) await loadConversation(conversations[0].id);

  byId('evaluate-btn').onclick = async () => {
    const judgeModelIds = [...byId('eval-judges').selectedOptions].map((o) => Number(o.value));
    const created = await api('/api/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        conversation_id: Number(byId('eval-conversation').value),
        main_model_id: Number(byId('eval-main-model').value),
        judge_model_ids: judgeModelIds,
      }),
    });
    const result = await api(`/api/evaluate/${created.eval_job_id}`);
    const scores = result.aggregate_report?.scores || {};
    const flags = result.aggregate_report?.flagged_lines || [];
    byId('eval-summary').innerHTML = `<table class="table table-sm"><thead><tr><th>Category</th><th>Score</th></tr></thead><tbody>${Object.entries(scores).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</tbody></table>`;
    byId('eval-flags').innerHTML = flags.map((f) => `<button class="list-group-item list-group-item-action" data-target="msg-${f.message_id}">${f.reason}: ${f.excerpt}</button>`).join('');
    byId('eval-flags').querySelectorAll('button').forEach((el) => {
      el.onclick = () => document.getElementById(el.dataset.target)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  };
}

(async function boot() {
  try {
    const page = document.body.dataset.page;
    if (page === 'setup') await initSetupPage();
    if (page === 'chat') await initChatPage();
    if (page === 'batch') await initBatchPage();
    if (page === 'evaluation') await initEvaluationPage();
  } catch (err) {
    alert(err.message);
  }
})();
