function $(id) { return document.getElementById(id); }

async function fetchJSON(url) {
  const res = await fetch(url);
  return await res.json();
}

async function refreshSensors() {
  try {
    const data = await fetchJSON('/api/sensors');
    const env = data.environment || {};
    const st = data.status || {};
    ['temperature_c','humidity_percent','pressure_hpa','gas_ohms','altitude_m'].forEach(k => {
      const el = $(k);
      if (el) el.textContent = env[k] == null ? '-' : env[k].toFixed ? env[k].toFixed(2) : env[k];
    });
    const offline = Object.entries(st).filter(([,v]) => v !== 'online').map(([k]) => k);
    $('env-status').textContent = offline.length ? `⚠ Offline: ${offline.join(', ')}` : 'All sensors online';
  } catch (e) {
    $('env-status').textContent = '⚠ Sensors unavailable';
  }
}

async function refreshSummary() {
  try {
    const sum = await fetchJSON('/api/summary');
    $('summary').textContent = JSON.stringify(sum, null, 2);
  } catch (e) {
    $('summary').textContent = 'No summary available yet';
  }
}

async function refreshAlerts() {
  try {
    const a = await fetchJSON('/api/alerts');
    $('alerts').textContent = JSON.stringify(a.alerts || [], null, 2);
  } catch (e) {
    $('alerts').textContent = 'No alerts available';
  }
}

document.addEventListener('DOMContentLoaded', () => {
  refreshSensors();
  refreshSummary();
  setInterval(refreshSensors, 4000);
  setInterval(refreshSummary, 10000);

  const alertsBtn = $('alertsBtn');
  const modalEl = document.getElementById('alertsModal');
  const modal = new bootstrap.Modal(modalEl);
  alertsBtn.addEventListener('click', async () => {
    await refreshAlerts();
    modal.show();
  });
});


