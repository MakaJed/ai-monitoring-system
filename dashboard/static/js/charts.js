(() => {
  // Charts are plain canvas renders (no external libs)
  async function loadSummaries() {
    try {
      const res = await fetch('/api/summaries');
      const data = await res.json();
      const daily = data.daily || [];
      const labels = daily.map(d => d.day);
      const temps = daily.map(d => (d.avg_temp ?? null));
      const hums  = daily.map(d => (d.avg_humidity ?? null));
      const gas   = daily.map(d => (d.avg_gas ?? null));
      const statusSummary = data.status_summary || {normal:0, warning:0, critical:0};
      const statusCounts = [statusSummary.normal||0, statusSummary.warning||0, statusSummary.critical||0];
      const symCounts = daily.map(d => (d.symptom_count || 0));

      // Trend + threshold colors
      const firstVal = arr => arr.find(v => v != null);
      const lastVal  = arr => { for (let i = arr.length - 1; i >= 0; i--) { if (arr[i] != null) return arr[i]; } return null; };

      const t0 = firstVal(temps); const t1 = lastVal(temps);
      let tempColor = '#059669'; // Muted green
      if (t1 != null && (t1 < 20 || t1 > 33)) tempColor = '#b91c1c'; // Muted red
      else if (t0 != null && t1 != null && t1 > t0) tempColor = '#d97706'; // Muted orange

      const h0 = firstVal(hums); const h1 = lastVal(hums);
      let humColor = '#059669'; // Muted green
      if (h1 != null && (h1 < 50 || h1 > 85)) humColor = '#b91c1c'; // Muted red
      else if (h0 != null && h1 != null && h1 > h0) humColor = '#d97706'; // Muted orange

      const g1 = lastVal(gas);
      let gasColor = '#0891b2'; // Muted blue
      if (g1 != null && g1 < 50) gasColor = '#b91c1c'; // Muted red
      else if (g1 != null && g1 < 100) gasColor = '#d97706'; // Muted orange

      drawLineChart('chart-temp', labels, temps, tempColor, 'Temperature Trend (°C)');
      drawDualAxis('chart-hg', labels, hums, gas, humColor, gasColor, 'Humidity (%)', 'Gas (kΩ)');
      drawDonut('chart-status', statusCounts, ['#059669','#d97706','#b91c1c'], ['Normal','Warning','Critical']);
      drawBarChart ('chart-sym', labels, symCounts,  '#7c3aed', 'Symptoms per Day');
    } catch (e) {
      const charts = document.getElementById('charts');
      charts.textContent = 'Failed to load summaries';
    }
  }

  function drawLineChart(canvasId, labels, data, color, title){
    const c = document.getElementById(canvasId); if(!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0,0,c.width,c.height);
    // title
    ctx.fillStyle = '#292524'; ctx.font = '13px -apple-system, sans-serif'; ctx.fillText(title, 8, 18);
    const pad = 30; const w = c.width - pad*2; const h = c.height - pad*2;
    const xs = (i)=> pad + (w * i / Math.max(1, labels.length-1));
    const vals = data.filter(v=>v!=null);
    const min = Math.min(...vals, 0); const max = Math.max(...vals, 1);
    const ys = (v)=> pad + h - (h * ((v - min)/Math.max(1e-6, (max-min))));
    // axis
    ctx.strokeStyle = '#d6d3d1'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,pad+h); ctx.lineTo(pad+w,pad+h); ctx.stroke();
    // line
    ctx.strokeStyle = color; ctx.beginPath();
    data.forEach((v,i)=>{ if(v==null) return; const x=xs(i), y=ys(v); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.stroke();
    // points (apply temperature thresholds coloring if applicable)
    data.forEach((v,i)=>{
      if(v==null) return; const x=xs(i), y=ys(v);
      let pc = color;
      if (title.toLowerCase().includes('temperature')) {
        if (v < 20 || v > 33) pc = '#b91c1c'; // Muted red
        else if (v >= 30 && v <= 33) pc = '#d97706'; // Muted orange
      }
      ctx.fillStyle = pc; ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2); ctx.fill();
    });
    // x labels (sparse)
    ctx.fillStyle = '#78716c'; ctx.font = '10px -apple-system, sans-serif';
    labels.forEach((lab,i)=>{ if(i%Math.ceil(labels.length/8)!==0) return; const x=xs(i); ctx.fillText(lab, x-10, pad+h+12); });
  }

  function drawBarChart(canvasId, labels, data, color, title){
    const c = document.getElementById(canvasId); if(!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0,0,c.width,c.height);
    ctx.fillStyle = '#292524'; ctx.font = '13px -apple-system, sans-serif'; ctx.fillText(title, 8, 18);
    const pad = 30; const w = c.width - pad*2; const h = c.height - pad*2;
    const max = Math.max(1, ...data);
    const bw = Math.max(2, Math.floor(w / Math.max(1, data.length)) - 2);
    // axis
    ctx.strokeStyle = '#d6d3d1'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,pad+h); ctx.lineTo(pad+w,pad+h); ctx.stroke();
    // bars
    ctx.fillStyle = color;
    data.forEach((v,i)=>{
      const x = pad + i*(bw+2);
      const bh = Math.round(h * (v/max));
      ctx.fillRect(x, pad+h-bh, bw, bh);
    });
    // labels sparse
    ctx.fillStyle = '#78716c'; ctx.font = '10px -apple-system, sans-serif';
    labels.forEach((lab,i)=>{ if(i%Math.ceil(labels.length/8)!==0) return; const x = pad + i*(bw+2); ctx.fillText(lab, x, pad+h+12); });
  }

  function drawDualAxis(canvasId, labels, dataL, dataR, colorL, colorR, labelL, labelR){
    const c = document.getElementById(canvasId); if(!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0,0,c.width,c.height);
    ctx.fillStyle = '#111827'; ctx.font = '14px sans-serif'; ctx.fillText(`${labelL} & ${labelR}`, 8, 18);
    const pad = 30; const w = c.width - pad*2; const h = c.height - pad*2;
    const xs = (i)=> pad + (w * i / Math.max(1, labels.length-1));
    const valsL = dataL.filter(v=>v!=null); const minL = Math.min(...valsL, 0); const maxL = Math.max(...valsL, 1);
    const valsR = dataR.filter(v=>v!=null); const minR = Math.min(...valsR, 0); const maxR = Math.max(...valsR, 1);
    const ysL = (v)=> pad + h - (h * ((v - minL)/Math.max(1e-6, (maxL-minL))));
    const ysR = (v)=> pad + h - (h * ((v - minR)/Math.max(1e-6, (maxR-minR))));
    // axes
    ctx.strokeStyle = '#d6d3d1'; ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,pad+h); ctx.lineTo(pad+w,pad+h); ctx.stroke();
    // left series
    ctx.strokeStyle = colorL; ctx.beginPath();
    dataL.forEach((v,i)=>{ if(v==null) return; const x=xs(i), y=ysL(v); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.stroke();
    // right series (dashed)
    ctx.setLineDash([5,3]); ctx.strokeStyle = colorR; ctx.beginPath();
    dataR.forEach((v,i)=>{ if(v==null) return; const x=xs(i), y=ysR(v); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.stroke(); ctx.setLineDash([]);
    // legend
    ctx.fillStyle = '#64748b'; ctx.font = '11px sans-serif';
    ctx.fillText(labelL, pad, pad-6); ctx.fillText(labelR, pad+120, pad-6);
  }

  function drawDonut(canvasId, values, colors, labels){
    const c = document.getElementById(canvasId); if(!c) return;
    const ctx = c.getContext('2d');
    ctx.clearRect(0,0,c.width,c.height);
    const cx = c.width/2, cy = c.height/2, r = Math.min(c.width,c.height)/2 - 20;
    const total = values.reduce((a,b)=>a+b,0) || 1;
    let start = -Math.PI/2;
    values.forEach((v,i)=>{
      const slice = (v/total) * Math.PI*2; const end = start + slice;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.fillStyle = colors[i] || '#ccc';
      ctx.arc(cx,cy, r, start, end); ctx.lineTo(cx,cy); ctx.fill(); start = end;
    });
    // inner cutout
    ctx.globalCompositeOperation = 'destination-out';
    ctx.beginPath(); ctx.arc(cx,cy, r*0.6, 0, Math.PI*2); ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
    // legend
    ctx.fillStyle = '#292524'; ctx.font = '12px -apple-system, sans-serif';
    let lx = 10, ly = 18;
    labels.forEach((lab,i)=>{ ctx.fillStyle = colors[i]; ctx.fillRect(lx, ly-10, 10, 10); ctx.fillStyle = '#57534e'; ctx.fillText(`${lab}: ${values[i]}`, lx+16, ly); ly += 16; });
  }

  loadSummaries();
})();
