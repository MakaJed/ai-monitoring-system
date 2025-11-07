(() => {
  const alertBanner = document.getElementById('alert-banner');
  const alertText = document.getElementById('alert-text');
  const alertClose = document.getElementById('alert-close');
  let lastAlertShownAt = 0;
  let dismissedUntil = 0;

  const streamVisible = document.getElementById('stream-visible');
  const streamThermal = document.getElementById('stream-thermal');

  const cpuPercent = document.getElementById('cpu-percent');
  const cpuTemp = document.getElementById('cpu-temp');
  const memPercent = document.getElementById('mem-percent');
  const uptime = document.getElementById('uptime');
  const onlineIndicator = document.getElementById('online-indicator');

  const tempEl = document.getElementById('temp');
  const humEl = document.getElementById('hum');
  const gasEl = document.getElementById('gas');
  const presEl = document.getElementById('pres');
  const badgeTemp = document.getElementById('badge-temp');
  const badgeHum = document.getElementById('badge-hum');
  const badgeGas = document.getElementById('badge-gas');
  const badgePres = document.getElementById('badge-pres');

  const notifBadge = document.getElementById('notif-badge');
  const notifToggle = document.getElementById('notif-toggle');
  const notifMenu = document.getElementById('notif-menu');
  const notifList = document.getElementById('notif-list');
  const notifCount = document.getElementById('notif-count');
  const notifMarkAll = document.getElementById('notif-mark-all');
  const chartArea = document.getElementById('chart-area');
  const modelBadge = document.getElementById('model-status');

  alertClose?.addEventListener('click', () => {
    dismissedUntil = Date.now();
    alertBanner.classList.add('hidden');
  });

  function humanUptime(seconds) {
    if (!seconds && seconds !== 0) return '-';
    const d = Math.floor(seconds / 86400);
    const h = Math.floor((seconds % 86400) / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${d}d ${h}h ${m}m`;
  }

  async function pollStatus() {
    try {
      const res = await fetch('/status');
      if (!res.ok) throw new Error('Status not OK');
      const data = await res.json();
      cpuPercent.textContent = data.cpu_percent ?? '-';
      cpuTemp.textContent = data.cpu_temp_c ?? '-';
      memPercent.textContent = data.memory_percent ?? '-';
      uptime.textContent = humanUptime(data.uptime_seconds);
      if (data.online) {
        onlineIndicator.classList.remove('offline');
        onlineIndicator.classList.add('online');
        onlineIndicator.textContent = 'Online';
      } else {
        onlineIndicator.classList.remove('online');
        onlineIndicator.classList.add('offline');
        onlineIndicator.textContent = 'Offline';
      }
    } catch (e) {
      // Offline or error - show offline state but don't break
      onlineIndicator.classList.remove('online');
      onlineIndicator.classList.add('offline');
      onlineIndicator.textContent = 'Offline';
      // Keep last known values, don't clear them
    }
  }

  function setBadge(el, value, warnThresh, alertThresh) {
    el.classList.remove('badge-ok', 'badge-warn', 'badge-alert');
    if (value == null || Number.isNaN(value)) {
      el.textContent = 'Offline';
      el.classList.add('badge-alert');
      return;
    }
    if (alertThresh != null && value >= alertThresh) {
      el.textContent = 'Alert';
      el.classList.add('badge-alert');
    } else if (warnThresh != null && value >= warnThresh) {
      el.textContent = 'Warning';
      el.classList.add('badge-warn');
    } else {
      el.textContent = 'Normal';
      el.classList.add('badge-ok');
    }
  }

  async function pollSensors() {
    try {
      const res = await fetch('/sensor_data');
      if (!res.ok) throw new Error('Sensor data not OK');
      const data = await res.json();
      if (data.status === 'offline') {
        tempEl.textContent = '-';
        humEl.textContent = '-';
        gasEl.textContent = '-';
        presEl.textContent = '-';
        setBadge(badgeTemp, null);
        setBadge(badgeHum, null);
        setBadge(badgeGas, null);
        setBadge(badgePres, null);
        return;
      }
      tempEl.textContent = data.temperature_c ?? '-';
      humEl.textContent = data.humidity_pct ?? '-';
      gasEl.textContent = data.gas_ohms ?? '-';
      presEl.textContent = data.pressure_hpa ?? '-';

      setBadge(badgeTemp, data.temperature_c, 30, 38);
      setBadge(badgeHum, data.humidity_pct, 70, 85);
      setBadge(badgeGas, data.gas_ohms, null, null);
      setBadge(badgePres, data.pressure_hpa, null, null);
    } catch (e) {
      // Offline or error - show offline badges but keep UI functional
      setBadge(badgeTemp, null);
      setBadge(badgeHum, null);
      setBadge(badgeGas, null);
      setBadge(badgePres, null);
    }
  }

  function renderNotifications(alerts){
    const unread = (alerts||[]).filter(a=>!a.read);
    notifBadge.textContent = unread.length;
    notifCount.textContent = `${alerts.length} total`;
    notifList.innerHTML = (alerts||[]).map(a=>{
      const cls = a.read ? 'notif-item' : 'notif-item unread';
      const t = (a.timestamp||'').replace('T',' ').replace('Z','');
      return `<li class="${cls}"><div>${a.text||''}</div><div class="notif-time">${t}</div></li>`;
    }).join('') || '<li class="notif-item">No notifications</li>';
  }

  async function pollNotifications() {
    try {
      const res = await fetch('/notifications');
      if (!res.ok) throw new Error('Notifications not OK');
      const data = await res.json();
      const alerts = data.alerts || [];
      renderNotifications(alerts);
      if (alerts.length > 0) {
        const latest = alerts[alerts.length - 1]?.text || '';
        if (Date.now() - dismissedUntil > 1000 && latest && latest !== alertText.textContent) {
          alertText.textContent = latest;
          alertBanner.classList.remove('hidden');
          lastAlertShownAt = Date.now();
        }
      }
    } catch (e) {
      // Offline or error - silently fail, keep last state
    }
  }

  notifToggle?.addEventListener('click', () => {
    notifMenu.style.display = (notifMenu.style.display === 'block') ? 'none' : 'block';
  });
  document.addEventListener('click', (e)=>{
    if (!notifMenu.contains(e.target) && e.target !== notifToggle) {
      notifMenu.style.display = 'none';
    }
  });
  notifMarkAll?.addEventListener('click', async ()=>{
    try{
      const res = await fetch('/notifications');
      const data = await res.json();
      const ids = (data.alerts||[]).map(a=>a.id);
      await fetch('/api/alerts/read', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ids})});
      pollNotifications();
    }catch(e){}
  });

  async function pollModelStatus() {
    try {
      const res = await fetch('/model_status');
      if (!res.ok) throw new Error('Model status not OK');
      const data = await res.json();
      modelBadge.classList.remove('badge-ok', 'badge-warn', 'badge-alert');
      if (data.available) {
        modelBadge.textContent = `Model: OK (${data.model}, ${data.labels})`;
        modelBadge.classList.add('badge-ok');
      } else {
        const reason = data.reason || 'unavailable';
        modelBadge.textContent = `Model: ${reason}`;
        modelBadge.classList.add('badge-alert');
      }
    } catch (e) {
      // Offline or error - show error state
      modelBadge.classList.remove('badge-ok');
      modelBadge.classList.add('badge-alert');
      modelBadge.textContent = 'Model: offline';
    }
  }

  async function loadAnalytics() {
    try {
      const res = await fetch('/api/summaries');
      if (!res.ok) throw new Error('Analytics not OK');
      const data = await res.json();
      const daily = data.daily || [];
      const today = daily[0] || {};
      const avgT = today.avg_temp ?? '-';
      const avgH = today.avg_humidity ?? '-';
      const abn  = today.abnormal ?? 0;
      if (chartArea) {
        chartArea.textContent = `Today — Avg Temp: ${avgT} °C | Avg Hum: ${avgH} % | Abnormal: ${abn}`;
      }
    } catch (e) {
      // Offline or error - show unavailable but don't break
      if (chartArea) {
        chartArea.textContent = 'Analytics unavailable';
      }
    }
  }

  async function tickDetect() {
    try { await fetch('/detect'); } catch (e) {}
  }

  // Init: always-on split view
  const ts = Date.now();
  streamVisible.src = `/camera_feed?ts=${ts}`;
  streamThermal.src = `/thermal_feed?ts=${ts}`;
  pollStatus();
  pollSensors();
  pollNotifications();
  pollModelStatus();
  loadAnalytics();
  tickDetect();

  setInterval(pollStatus, 5000);
  setInterval(pollSensors, 5000);
  setInterval(pollNotifications, 5000);
  setInterval(pollModelStatus, 5000);
  setInterval(loadAnalytics, 30000);
  setInterval(tickDetect, 2000);
})();
