const api = async (path, opts = {}) => {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const payload = await res.json();
  if (!res.ok || payload.success === false) {
    throw new Error(payload.error || `Request failed: ${res.status}`);
  }
  return payload.data;
};

const statusDot = (status) => `<span class="status-dot status-${status || 'red'}"></span>`;
const byId = (id) => document.getElementById(id);

async function initSetupPage() {
  const modelModal = new bootstrap.Modal(byId('model-modal'));
  const agentModal = new bootstrap.Modal(byId('agent-modal'));

  const loadModels = async () => {
    const models = await api('/api/models');
    byId('model-list').innerHTML = models.map((m) => `
      <div class="list-group-item d-flex justify-content-between align-items-center">
        <div>${statusDot(m.status)}<strong>${m.name}</strong> <small class="text-muted">${m.host}:${m.port} • ${m.backend}/${m.model_name}</small></div>
        <div>
          <button class="btn btn-sm btn-outline-secondary" onclick='window.editModel(${JSON.stringify(m)})'>Edit</button>
          <button class="btn btn-sm btn-outline-danger" onclick='window.deleteModel(${m.id})'>Delete</button>
        </div>
      </div>
    `).join('');
    const select = byId('agent-model-id');
    select.innerHTML = models.map((m) => `<option value="${m.id}">${m.name} (${m.model_name})</option>`).join('');
  };

  const loadAgents = async () => {
    const agents = await api('/api/agents');
    byId('agent-list').innerHTML = agents.map((a) => `
      <div class="list-group-item d-flex justify-content-between align-items-center">
        <div>${statusDot(a.effective_status || 'yellow')}<strong>${a.name}</strong> <small class="text-muted">${a.model_name || a.model_id}</small></div>
        <div>
          <button class="btn btn-sm btn-outline-secondary" onclick='window.editAgent(${JSON.stringify(a)})'>Edit</button>
          <button class="btn btn-sm btn-outline-danger" onclick='window.deleteAgent(${a.id})'>Delete</button>
        </div>
      </div>
    `).join('');
  };

  byId('add-model-btn').onclick = () => { byId('model-form').reset(); byId('model-id').value = ''; modelModal.show(); };
  byId('add-agent-btn').onclick = () => { byId('agent-form').reset(); byId('agent-id').value = ''; agentModal.show(); };
  byId('refresh-models').onclick = async () => { await api('/api/models/probe', { method: 'POST' }); await loadModels(); await loadAgents(); };

  window.editModel = (m) => {
    byId('model-id').value = m.id; byId('model-name').value = m.name; byId('model-host').value = m.host;
    byId('model-port').value = m.port; byId('model-backend').value = m.backend; byId('model-model-name').value = m.model_name; modelModal.show();
  };
  window.deleteModel = async (id) => { if (confirm('Delete model?')) { await api(`/api/models/${id}`, { method: 'DELETE' }); await loadModels(); await loadAgents(); } };

  window.editAgent = (a) => {
    byId('agent-id').value = a.id; byId('agent-name').value = a.name; byId('agent-model-id').value = a.model_id;
    byId('agent-system-prompt').value = a.system_prompt; byId('agent-max-tokens').value = a.max_tokens; byId('agent-temperature').value = a.temperature; agentModal.show();
  };
  window.deleteAgent = async (id) => { if (confirm('Delete agent?')) { await api(`/api/agents/${id}`, { method: 'DELETE' }); await loadAgents(); } };

  byId('model-form').onsubmit = async (e) => {
    e.preventDefault();
    const id = byId('model-id').value;
    const body = {
      name: byId('model-name').value, host: byId('model-host').value, port: Number(byId('model-port').value),
      backend: byId('model-backend').value, model_name: byId('model-model-name').value,
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
  const agents = await api('/api/agents');
  const alive = agents.filter((a) => a.effective_status === 'green');
  const options = alive.map((a) => `<option value="${a.id}">${a.name}</option>`).join('');
  byId('chat-agent1').innerHTML = options;
  byId('chat-agent2').innerHTML = options;

  const pane = byId('chat-pane');
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

  byId('start-chat').onclick = () => {
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
  const agents = await api('/api/agents');
  const alive = agents.filter((a) => a.effective_status === 'green');
  const options = alive.map((a) => `<option value="${a.id}">${a.name}</option>`).join('');
  byId('batch-agent1').innerHTML = options;
  byId('batch-agent2').innerHTML = options;

  let activeBatchId = null;

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

  byId('start-batch').onclick = async () => {
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
