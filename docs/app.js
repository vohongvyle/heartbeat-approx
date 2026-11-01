// Arrhythmia Approximation Lab: runs the 8-bit network in the browser from data.js
(async function () {
  const D = window.ECG, C = D.circuits, NL = D.layers.length, N = D.n_test, CL = D.classes;
  const $ = id => document.getElementById(id);
  const EX = C.findIndex(c => c.name === "mul8u_1JFF");
  const short = n => n.replace("mul8u_", "");
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  if (D.data !== "synthetic") $("banner").hidden = true;

  // ---------- unpack
  const bin = Uint8Array.from(atob(window.LUTS_GZ), ch => ch.charCodeAt(0));
  const LUT = new Uint16Array(await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
  const xb = Uint8Array.from(atob(D.x_b64), ch => ch.charCodeAt(0));
  const XQ = new Int16Array(xb.buffer);                       // layer-1 input, signed 8-bit magnitudes
  const L = D.layers.map(l => ({ ...l, mag: Int32Array.from(l.w, Math.abs), sgn: Int8Array.from(l.w, v => v < 0 ? -1 : 1), k: l.sa * l.sw }));
  const rhe = v => { const f = Math.floor(v), d = v - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); }; // numpy rounding
  const q8 = (v, s) => Math.min(255, rhe(Math.abs(v) / s));

  // ---------- 8-bit inference, mirrors cnn.QuantCNN.logits
  function dense(l, inMag, inSgn, R, c, comp, out, outOff) {
    const base = c * 65536, cols = l.cols;
    for (let o = 0; o < cols; o++) {
      let acc = 0;
      for (let r = 0; r < R; r++) { const w = r * cols + o; acc += inSgn[r] * l.sgn[w] * LUT[base + (inMag[r] << 8) + l.mag[w]]; }
      out[outOff + o] = (comp ? acc - l.corr[c][o] : acc) * l.k + l.b[o];
    }
  }
  function forward(cfg) {
    const [l1, l2, l3, l4] = L, out = new Float64Array(N * 4);
    const pm = new Int32Array(40), ps = new Int8Array(40).fill(1), z = new Float64Array(32);
    const a1 = new Float64Array(61 * 8), m1 = new Float64Array(30 * 8), a2 = new Float64Array(26 * 16);
    const fMag = new Int32Array(210), fSgn = new Int8Array(210), hMag = new Int32Array(32), hSgn = new Int8Array(32).fill(1);
    const m1q = new Int32Array(30 * 8);
    for (let n = 0; n < N; n++) {
      const xo = n * 128;
      for (let p = 0; p < 61; p++) {                           // conv1: 7 taps, stride 2
        for (let k = 0; k < 7; k++) { const v = XQ[xo + 2 * p + k]; pm[k] = v < 0 ? -v : v; ps[k] = v < 0 ? -1 : 1; }
        dense(l1, pm, ps, 7, cfg[0][0], cfg[0][1], a1, p * 8);
      }
      for (let p = 0; p < 30; p++) for (let ch = 0; ch < 8; ch++)
        m1[p * 8 + ch] = Math.max(0, Math.max(a1[(2 * p) * 8 + ch], a1[(2 * p + 1) * 8 + ch]));
      for (let i = 0; i < 240; i++) m1q[i] = q8(m1[i], l2.sa);
      ps.fill(1);
      for (let p = 0; p < 26; p++) {                           // conv2: 5 taps x 8 channels
        for (let k = 0; k < 5; k++) for (let ch = 0; ch < 8; ch++) pm[k * 8 + ch] = m1q[(p + k) * 8 + ch];
        dense(l2, pm, ps, 40, cfg[1][0], cfg[1][1], a2, p * 16);
      }
      for (let p = 0; p < 13; p++) for (let ch = 0; ch < 16; ch++) {
        const v = Math.max(0, Math.max(a2[(2 * p) * 16 + ch], a2[(2 * p + 1) * 16 + ch]));
        fMag[p * 16 + ch] = q8(v, l3.sa); fSgn[p * 16 + ch] = 1;
      }
      for (let j = 0; j < 2; j++) { const v = D.rr[n * 2 + j]; fMag[208 + j] = q8(v, l3.sa); fSgn[208 + j] = v < 0 ? -1 : 1; }
      dense(l3, fMag, fSgn, 210, cfg[2][0], cfg[2][1], z, 0);
      for (let o = 0; o < 32; o++) hMag[o] = q8(Math.max(0, z[o]), l4.sa);
      dense(l4, hMag, hSgn, 32, cfg[3][0], cfg[3][1], out, n * 4);
    }
    return out;
  }
  const cache = new Map();
  function run(cfg) {
    const key = cfg.map(x => x[0] + (x[1] ? "c" : "")).join("-");
    if (!cache.has(key)) {
      const lg = forward(cfg), pred = new Uint8Array(N), conf = new Float64Array(N);
      for (let n = 0; n < N; n++) {
        let b = 0; for (let j = 1; j < 4; j++) if (lg[n * 4 + j] > lg[n * 4 + b]) b = j;
        let s = 0; for (let j = 0; j < 4; j++) s += Math.exp(lg[n * 4 + j] - lg[n * 4 + b]);
        pred[n] = b; conf[n] = 1 / s;
      }
      cache.set(key, { pred, conf });
    }
    return cache.get(key);
  }
  function metrics(pred) {
    const cm = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
    for (let n = 0; n < N; n++) cm[D.y[n]][pred[n]]++;
    const tot = cm.map(r => r.reduce((a, b) => a + b, 0));
    const se = cm.map((r, i) => tot[i] ? r[i] / tot[i] : NaN);
    let ok = 0, miss = 0; for (let i = 0; i < 4; i++) { ok += cm[i][i]; if (i) miss += cm[i][0]; }
    return { cm, se, acc: ok / N, miss };
  }
  const energy = cfg => {
    let e = 0, e0 = 0;
    D.macs.forEach((m, i) => { e += m * (C[cfg[i][0]].pwr + D.adder); e0 += m * (C[EX].pwr + D.adder); });
    return e / e0;
  };

  // ---------- state + controls
  const exactCfg = Array.from({ length: NL }, () => [EX, false]);
  let cfg = D.presets.accuracy.map(x => [...x]);
  try { const s = JSON.parse(localStorage.getItem("aal-cfg") || "null"); if (s && s.length === NL && s.every(x => C[x[0]])) cfg = s; } catch (e) {}
  const TAUS = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.01];

  const names = ["Conv 1 · 7 taps", "Conv 2 · 5 taps", "Dense 1 · 210 → 32", "Dense 2 · 32 → 4"];
  const opts = C.map((c, i) => `<option value="${i}">${short(c.name)}${i === EX ? " (exact)" : ""} · ${c.pwr.toFixed(3)} mW · MAE ${c.mae}%</option>`).join("");
  $("layers").innerHTML = L.map((l, i) => `<div class="layer"><label for="s${i}">${names[i]}<span>${D.macs[i].toLocaleString()} mults</span></label>
    <select id="s${i}">${opts}</select><label class="chk" for="k${i}"><input type="checkbox" id="k${i}"> bias compensation</label></div>`).join("");
  L.forEach((l, i) => {
    $("s" + i).addEventListener("change", e => { cfg[i] = [+e.target.value, cfg[i][1]]; update(); });
    $("k" + i).addEventListener("change", e => { cfg[i] = [cfg[i][0], e.target.checked]; update(); });
  });
  const idx = n => C.findIndex(c => c.name === n);
  const presets = [
    ["Exact", exactCfg],
    ["Accuracy search", D.presets.accuracy],
    ["Clinical search", D.presets.clinical],
    ["1AGV everywhere", Array(NL).fill([idx("mul8u_1AGV"), false])],
    ["JV3 everywhere", Array(NL).fill([idx("mul8u_JV3"), false]), "warn"],
  ];
  $("presets").innerHTML = presets.map(([l, , w], i) => `<button class="chip ${w || ""}" data-i="${i}">${l}</button>`).join("");
  $("presets").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; cfg = presets[+b.dataset.i][1].map(x => [...x]); update(); });

  // ---------- rendering
  let cur = null, beat = -1;
  const base = metrics(run(exactCfg).pred);
  function setD(id, cur, ref, worseIsLower = true, fmt = v => (v * 100).toFixed(1)) {
    const d = cur - ref, el = $(id);
    el.textContent = Math.abs(d) < 1e-9 ? "same as exact" : `${d > 0 ? "+" : "−"}${fmt(Math.abs(d))} vs exact`;
    el.className = "d " + (Math.abs(d) < 1e-9 ? "" : ((d < 0) === worseIsLower ? "bad" : "good"));
    return d;
  }
  function renderVitals(m, e) {
    $("vE").textContent = pct(e, 0);
    $("vAcc").textContent = pct(m.acc); setD("dAcc", m.acc, base.acc);
    [["S", 1], ["V", 2], ["F", 3]].forEach(([k, i]) => {
      $("v" + k).textContent = pct(m.se[i]);
      const d = setD("d" + k, m.se[i], base.se[i]);
      $("t" + k).classList.toggle("flag", d < -0.01);
    });
    $("vM").textContent = m.miss;
    const d = m.miss - base.miss, el = $("dM");
    el.textContent = d === 0 ? "same as exact" : `${d > 0 ? "+" : "−"}${Math.abs(d)} vs exact`;
    el.className = "d " + (d > 0 ? "bad" : d < 0 ? "good" : "");
    $("tM").classList.toggle("flag", d > 0);
  }
  function renderCM(m) {
    let h = `<div class="h"></div>` + CL.map(c => `<div class="h">→ ${c}</div>`).join("");
    for (let i = 0; i < 4; i++) {
      h += `<div class="h" style="text-align:right">${CL[i]}</div>`;
      for (let j = 0; j < 4; j++) {
        const v = m.cm[i][j], d = v - base.cm[i][j];
        const better = (i === j) ? d > 0 : d < 0, cls = ["c", i === j ? "diag" : "", d === 0 ? "" : better ? "better" : "worse", (i > 0 && j === 0 && v > 0) ? "miss" : ""].join(" ");
        h += `<div class="${cls}">${v}<small>${d === 0 ? "·" : (d > 0 ? "+" : "") + d}</small></div>`;
      }
    }
    $("cm").innerHTML = h;
  }

  function gate(r, ex, tau) {
    const pred = new Uint8Array(N); let k = 0;
    for (let n = 0; n < N; n++) { const redo = r.pred[n] === 0 && r.conf[n] < tau; if (redo) k++; pred[n] = redo ? ex.pred[n] : r.pred[n]; }
    return { pred, frac: k / N };
  }
  function renderNet(r, e) {
    const ex = run(exactCfg), ti = +$("tau").value, tau = TAUS[ti];
    $("tauV").textContent = tau > 1 ? "1 (always)" : tau.toFixed(tau >= 0.99 ? 3 : 2);
    const pts = TAUS.map(t => { const g = gate(r, ex, t); return { t, e: e + g.frac, frac: g.frac, miss: metrics(g.pred).miss }; });
    const p = pts[ti];
    $("nR").textContent = pct(p.frac); $("nE").textContent = pct(p.e, 0); $("nM").textContent = p.miss;
    $("nM0").textContent = `(exact alone ${base.miss})`;
    // chart
    const W = 520, H = 220, ml = 42, mr = 14, mt = 14, mb = 34;
    const xMax = Math.max(1.05, ...pts.map(q => q.e)), yMax = Math.max(base.miss, ...pts.map(q => q.miss)) * 1.15 + 1;
    const sx = v => ml + v / xMax * (W - ml - mr), sy = v => mt + (1 - v / yMax) * (H - mt - mb);
    let g = "";
    const yt = niceTicks(yMax);
    yt.forEach(v => g += `<line class="gl" x1="${ml}" x2="${W - mr}" y1="${sy(v)}" y2="${sy(v)}"/><text x="${ml - 6}" y="${sy(v) + 4}" text-anchor="end">${v}</text>`);
    for (let v = 0; v <= xMax + 1e-9; v += 0.25) g += `<text x="${sx(v)}" y="${H - mb + 16}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    g += `<text x="${(ml + W - mr) / 2}" y="${H - 3}" text-anchor="middle">energy per beat incl. re-checks</text>`;
    g += `<line x1="${ml}" x2="${W - mr}" y1="${sy(base.miss)}" y2="${sy(base.miss)}" stroke="${css("--exact")}" stroke-dasharray="4 4"/>`;
    g += `<polyline fill="none" stroke="${css("--approx")}" stroke-width="2" stroke-linejoin="round" points="${pts.map(q => sx(q.e) + "," + sy(q.miss)).join(" ")}"/>`;
    pts.forEach((q, i) => g += `<circle cx="${sx(q.e)}" cy="${sy(q.miss)}" r="${i === ti ? 7 : 4}" fill="${i === ti ? css("--panel") : css("--approx")}" stroke="${css("--approx")}" stroke-width="2"/>`);
    $("netChart").innerHTML = g;
  }
  function niceTicks(max) { const step = max > 60 ? 20 : max > 30 ? 10 : 5; const t = []; for (let v = 0; v <= max; v += step) t.push(v); return t; }

  // sweep chart (precomputed uniform configurations from the Python run)
  const SW = 520, SH = 300, sml = 40, smr = 12, smt = 10, smb = 34, YMIN = 40;
  const ssx = v => sml + v / 1.03 * (SW - sml - smr), ssy = v => smt + (100 - Math.max(YMIN, v)) / (100 - YMIN) * (SH - smt - smb);
  function renderSweep(m, e) {
    let g = "";
    for (let v = YMIN; v <= 100; v += 10) g += `<line class="gl" x1="${sml}" x2="${SW - smr}" y1="${ssy(v)}" y2="${ssy(v)}"/><text x="${sml - 6}" y="${ssy(v) + 4}" text-anchor="end">${v}</text>`;
    for (let v = 0; v <= 1.0001; v += 0.2) g += `<text x="${ssx(v)}" y="${SH - smb + 16}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    g += `<text x="${(sml + SW - smr) / 2}" y="${SH - 3}" text-anchor="middle">energy per beat, same circuit in all layers</text>`;
    const surf = css("--panel");
    D.sweep.forEach((r, i) => {
      g += `<circle class="pt" data-i="${i}" cx="${ssx(r.energy)}" cy="${ssy(100 * r.acc)}" r="4" fill="${css("--muted")}" stroke="${surf}" stroke-width="1.5"/>`;
      g += `<circle class="pt" data-i="${i}" cx="${ssx(r.energy)}" cy="${ssy(100 * r.se_S)}" r="4" fill="${css("--approx")}" stroke="${surf}" stroke-width="1.5"/>`;
      g += `<rect class="pt" data-i="${i}" x="${ssx(r.energy) - 3.5}" y="${ssy(100 * r.se_V) - 3.5}" width="7" height="7" fill="${css("--exact")}" stroke="${surf}" stroke-width="1.5"/>`;
    });
    [m.acc, m.se[1], m.se[2]].forEach(v => g += `<circle cx="${ssx(e)}" cy="${ssy(100 * v)}" r="8" fill="none" stroke="${css("--ink")}" stroke-width="1.8"/>`);
    $("sweep").innerHTML = g;
  }
  $("sweep").addEventListener("pointermove", ev => {
    const t = ev.target.closest(".pt"), tip = $("tip");
    if (!t) { tip.hidden = true; return; }
    const r = D.sweep[+t.dataset.i], box = $("sweep").getBoundingClientRect(), k = box.width / SW;
    tip.innerHTML = `${r.circuit} · energy ${pct(r.energy, 0)}<br>accuracy ${pct(r.acc)} · Se S ${pct(r.se_S)} · Se V ${pct(r.se_V)}`;
    tip.style.left = Math.min(box.width - 120, Math.max(120, ssx(r.energy) * k)) + "px"; tip.style.top = (+t.getAttribute("cy") || +t.getAttribute("y")) * k + "px"; tip.hidden = false;
  });
  $("sweep").addEventListener("pointerleave", () => $("tip").hidden = true);
  $("sweep").addEventListener("click", ev => { const t = ev.target.closest(".pt"); if (!t) return; const c = idx(D.sweep[+t.dataset.i].circuit); cfg = Array.from({ length: NL }, () => [c, false]); update(); });

  // beat on ECG paper
  function renderBeat() {
    const ex = run(exactCfg), r = cur;
    if (beat < 0) { $("paper").innerHTML = ""; return; }
    const W = 640, H = 300, s = L[0].sa;
    const sx = i => 10 + i / 127 * (W - 20);
    // 25 mm/s, 10 mm/mV: 0.7 s = 17.8 mm wide; draw squares in that scale
    const mmX = (W - 20) / (128 / 180 * 25), mid = H * 0.55, mmY = mmX;
    let g = `<rect width="${W}" height="${H}" fill="${css("--bg")}"/>`;
    for (let x = 10, i = 0; x <= W - 10 + 0.1; x += mmX, i++) g += `<line x1="${x}" x2="${x}" y1="0" y2="${H}" stroke="${css(i % 5 ? "--grid-minor" : "--grid-major")}" stroke-width="${i % 5 ? 0.6 : 1.1}"/>`;
    for (let k = -30; k <= 30; k++) { const y = mid + k * mmY; if (y < 0 || y > H) continue; g += `<line x1="0" x2="${W}" y1="${y}" y2="${y}" stroke="${css(k % 5 ? "--grid-minor" : "--grid-major")}" stroke-width="${k % 5 ? 0.6 : 1.1}"/>`; }
    const vals = Array.from({ length: 128 }, (_, i) => XQ[beat * 128 + i] * s);
    const amp = Math.max(...vals.map(Math.abs)) || 1, gain = (H * 0.4) / amp;
    g += `<polyline fill="none" stroke="${css("--trace")}" stroke-width="2.2" stroke-linejoin="round" points="${vals.map((v, i) => sx(i) + "," + (mid - v * gain)).join(" ")}"/>`;
    g += `<line x1="${sx(45)}" x2="${sx(45)}" y1="10" y2="22" stroke="${css("--muted")}"/><text x="${sx(45) + 4}" y="20">R peak</text>`;
    const pre = D.rr[beat * 2], post = D.rr[beat * 2 + 1];
    g += `<text x="${W - 12}" y="${H - 10}" text-anchor="end">25 mm/s · beat #${beat} · rhythm features ${pre.toFixed(2)}, ${post.toFixed(2)} (standardized)</text>`;
    $("paper").innerHTML = g;
    const yt = D.y[beat], pe = ex.pred[beat], pa = r.pred[beat];
    $("cT").textContent = CL[yt]; $("cE").textContent = CL[pe]; $("cA").textContent = CL[pa];
    $("cAw").classList.toggle("miss", yt !== 0 && pa === 0);
    $("cE").parentElement.classList.toggle("miss", yt !== 0 && pe === 0);
    $("beatNote").textContent = `Your chip's confidence: ${pct(r.conf[beat], 0)} for ${CL[pa]}. ` +
      (yt !== 0 && pa === 0 ? (pe !== 0 ? "The exact chip catches this one; your chip misses it." : "Both chips miss this beat.") :
        pa === yt ? "Your chip gets this one right." : `Wrong class, but it is still flagged as abnormal.`);
  }
  function pickBeat(kind) {
    const ex = run(exactCfg), r = cur, list = [];
    for (let n = 0; n < N; n++) {
      if (kind === "miss" && D.y[n] !== 0 && r.pred[n] === 0 && ex.pred[n] !== 0) list.push(n);
      else if (kind === "ab" && D.y[n] !== 0) list.push(n);
      else if (kind === "any") list.push(n);
    }
    if (!list.length) { $("beatNote").textContent = "No abnormal beat is missed by your chip but caught by the exact one. Try a cheaper configuration."; return; }
    beat = list[Math.floor(Math.random() * list.length)]; renderBeat();
  }
  $("bMiss").onclick = () => pickBeat("miss"); $("bAb").onclick = () => pickBeat("ab"); $("bAny").onclick = () => pickBeat("any");
  $("tau").addEventListener("input", () => { if (cur) renderNet(cur, energy(cfg)); });

  function update() {
    L.forEach((l, i) => { $("s" + i).value = cfg[i][0]; $("k" + i).checked = cfg[i][1]; });
    document.querySelectorAll("#presets button").forEach(b => b.setAttribute("aria-pressed", JSON.stringify(presets[+b.dataset.i][1]) === JSON.stringify(cfg)));
    try { localStorage.setItem("aal-cfg", JSON.stringify(cfg)); } catch (e) {}
    $("vitals").classList.add("busy");
    setTimeout(() => {
      cur = run(cfg); const m = metrics(cur.pred), e = energy(cfg);
      renderVitals(m, e); renderCM(m); renderNet(cur, e); renderSweep(m, e);
      if (beat < 0) pickBeat("miss"); if (beat < 0) pickBeat("ab"); else renderBeat();
      $("vitals").classList.remove("busy");
    }, 20);
  }
  update();
  const rerender = () => { if (!cur) return; const m = metrics(cur.pred), e = energy(cfg); renderNet(cur, e); renderSweep(m, e); renderBeat(); };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
  new MutationObserver(rerender).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // self-check against the Python run (console only)
  window.__parity = () => {
    const a = run(exactCfg).pred, b = run(D.presets.accuracy).pred;
    let d1 = 0, d2 = 0; for (let n = 0; n < N; n++) { d1 += a[n] !== D.ref.exact[n]; d2 += b[n] !== D.ref.accuracy[n]; }
    return { exact_mismatch: d1, accuracy_mismatch: d2, n: N };
  };
})();
