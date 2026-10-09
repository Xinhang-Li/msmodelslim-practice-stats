# -*- coding: utf-8 -*-
"""从 dash_data.json 生成单文件 HTML 看板（无外部依赖，可离线打开）"""
import os, json

WORK = os.environ.get('MSLIM_WORK', os.path.join(os.getcwd(), 'work'))
OUT = os.environ.get('MSLIM_OUT', os.getcwd())
REPO = os.environ.get('MSLIM_REPO', os.path.join(WORK, 'msmodelslim'))

data = json.load(open(os.path.join(WORK, 'dash_data.json')))

# 内嵌全部 YAML 原文：点击模型即可离线查看对应量化配置
yamls = {}
lp = os.path.join(REPO, 'lab_practice')
for r in data['rows']:
    y = r['yaml']
    if y in yamls: continue
    p = os.path.join(lp, y)
    if os.path.exists(p):
        yamls[y] = open(p, encoding='utf-8', errors='replace').read()
data['yamls'] = yamls

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>msModelSlim 最佳实践回退率看板</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#050810; --panel:#0c1220; --panel2:#111a2b; --line:#1a2540;
  --cyan:#22d3ee; --mint:#34d399; --amber:#fbbf24; --red:#f87171; --magenta:#e879f9;
  --tx:#e5eaf3; --tx2:#8b96ad; --tx3:#566179;
  --mono:'JetBrains Mono',ui-monospace,monospace;
  --sans:'Inter','PingFang SC','Microsoft YaHei',sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
html{background:var(--bg)}
body{
  background:
    radial-gradient(1200px 500px at 80% -10%, rgba(34,211,238,.06), transparent 60%),
    radial-gradient(900px 500px at 10% 110%, rgba(232,121,249,.04), transparent 60%),
    var(--bg);
  color:var(--tx); font-family:var(--sans); font-size:14px; min-height:100vh;
}
body::after{ /* scanline 纹理 */
  content:''; position:fixed; inset:0; pointer-events:none; z-index:60;
  background:repeating-linear-gradient(0deg, rgba(255,255,255,.012) 0 1px, transparent 1px 3px);
}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-thumb{background:var(--line);border-radius:3px}
::-webkit-scrollbar-track{background:transparent}
a{color:var(--cyan);text-decoration:none}

/* ---------- header ---------- */
header{
  position:sticky; top:0; z-index:50; display:flex; align-items:center; gap:14px;
  padding:10px 20px; background:rgba(5,8,16,.85); backdrop-filter:blur(8px);
  border-bottom:1px solid var(--line);
}
.lights{display:flex;gap:6px}
.lights i{width:10px;height:10px;border-radius:50%;background:var(--line)}
.lights i:nth-child(1){background:var(--red)} .lights i:nth-child(2){background:var(--amber)} .lights i:nth-child(3){background:var(--mint)}
header h1{font-size:15px;font-weight:800;letter-spacing:.02em}
header h1 span{color:var(--tx2);font-weight:400}
.live{
  margin-left:auto; display:flex; align-items:center; gap:8px;
  font-family:var(--mono); font-size:10px; letter-spacing:.18em; color:var(--mint);
  text-transform:uppercase;
}
.live i{width:7px;height:7px;border-radius:50%;background:var(--mint);animation:pulse 1.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(52,211,153,.5)}50%{opacity:.5;box-shadow:0 0 0 5px rgba(52,211,153,0)}}
.live b{color:var(--tx2);font-weight:400;letter-spacing:.05em;text-transform:none}

/* ---------- layout ---------- */
main{max-width:1560px;margin:0 auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.panel > .phead{
  display:flex;align-items:baseline;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line);
}
.phead h2{font-size:12px;font-weight:700;letter-spacing:.16em;color:var(--tx);text-transform:uppercase;font-family:var(--mono);white-space:nowrap}
.phead .sub{font-size:11px;color:var(--tx3)}
.pbody{padding:16px}

/* ---------- KPI ---------- */
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.kpi{background:var(--panel);padding:16px 18px;position:relative}
.kpi .k-label{font-family:var(--mono);font-size:9px;letter-spacing:.2em;color:var(--tx3);text-transform:uppercase;margin-bottom:8px}
.kpi .k-val{font-family:var(--mono);font-size:30px;font-weight:700;line-height:1;font-variant-numeric:tabular-nums}
.kpi .k-unit{font-size:12px;color:var(--tx2);font-weight:400;margin-left:3px}
.kpi .k-sub{font-size:10px;color:var(--tx3);margin-top:6px}
.kpi.c-cyan .k-val{color:var(--cyan)} .kpi.c-mint .k-val{color:var(--mint)}
.kpi.c-amber .k-val{color:var(--amber)} .kpi.c-red .k-val{color:var(--red)}
.kpi.c-magenta .k-val{color:var(--magenta)}

/* ---------- charts ---------- */
.grid3{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:16px}
.legend{display:flex;gap:14px;font-family:var(--mono);font-size:9px;letter-spacing:.12em;color:var(--tx2)}
.legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;vertical-align:-1px}
svg text{font-family:var(--mono)}
.bar-row{cursor:default}
.bar-row:hover .bar-bg{fill:rgba(255,255,255,.04)}
.tip{
  position:fixed;z-index:70;pointer-events:none;background:var(--panel2);border:1px solid var(--line);
  border-radius:6px;padding:8px 10px;font-family:var(--mono);font-size:11px;line-height:1.7;color:var(--tx);
  display:none;box-shadow:0 8px 24px rgba(0,0,0,.5);max-width:360px;
}
.tip .t-m{color:var(--tx2)}

/* ---------- table ---------- */
.ttools{display:flex;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
.grp{display:flex;align-items:center;gap:6px;font-size:12px;color:#8b96ad;cursor:pointer;user-select:none;font-family:'JetBrains Mono',monospace}
.grp input{accent-color:#22d3ee;cursor:pointer}
.g-head td{background:#111a2b;color:#22d3ee;font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.08em;padding:7px 12px;border-top:1px solid #1a2540;cursor:default}
.g-head .g-n{color:#8b96ad;margin-left:12px;letter-spacing:0}
.g-head .g-r{color:#fbbf24;margin-left:12px;letter-spacing:0}
.ttools input[type=text]{
  background:var(--bg);border:1px solid var(--line);border-radius:6px;color:var(--tx);
  font-family:var(--mono);font-size:11px;padding:6px 10px;width:260px;outline:none;
}
.ttools input:focus{border-color:var(--cyan);box-shadow:0 0 0 2px rgba(34,211,238,.15)}
.ttools .cnt{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--tx3);letter-spacing:.1em}
.twrap{max-height:640px;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:11.5px}
thead th{
  position:sticky;top:0;background:var(--panel2);z-index:5;text-align:left;
  font-family:var(--mono);font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--tx2);
  padding:9px 10px;border-bottom:1px solid var(--line);cursor:pointer;user-select:none;white-space:nowrap;
}
thead th:hover{color:var(--cyan)}
thead th .arr{color:var(--cyan);margin-left:3px}
tbody tr{border-bottom:1px solid rgba(26,37,64,.6);transition:opacity .15s ease;animation:rowIn .3s ease both}
tbody:hover tr{opacity:.15}
tbody tr:hover{opacity:1;background:rgba(34,211,238,.05)}
@keyframes rowIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
td{padding:7px 10px;vertical-align:top;color:var(--tx2)}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap;color:var(--tx)}
td.model{color:var(--tx);font-weight:600;white-space:nowrap}
td.yaml{font-family:var(--mono);font-size:10px;color:var(--tx3);max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
td .fp{color:var(--red)} td .up{color:var(--amber)} td .ok{color:var(--mint)}
td.note{font-size:10.5px;color:var(--tx3);max-width:260px}
.badge{display:inline-block;font-family:var(--mono);font-size:9px;padding:1px 6px;border-radius:3px;border:1px solid var(--line);color:var(--tx2);white-space:nowrap}
footer{padding:14px 20px 30px;color:var(--tx3);font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-align:center}

/* ---------- YAML 抽屉 ---------- */
.backdrop{position:fixed;inset:0;background:rgba(2,4,10,.6);backdrop-filter:blur(2px);z-index:80;opacity:0;pointer-events:none;transition:opacity .25s ease}
.backdrop.show{opacity:1;pointer-events:auto}
.drawer{
  position:fixed;top:0;right:0;bottom:0;width:min(620px,92vw);z-index:90;
  background:var(--panel);border-left:1px solid var(--line);box-shadow:-16px 0 48px rgba(0,0,0,.5);
  transform:translateX(102%);transition:transform .32s cubic-bezier(.22,1,.36,1);
  display:flex;flex-direction:column;
}
.drawer.show{transform:none}
.d-head{display:flex;align-items:center;gap:10px;padding:14px 18px;border-bottom:1px solid var(--line);background:var(--panel2)}
.d-head .d-dot{width:8px;height:8px;border-radius:50%;background:var(--cyan);box-shadow:0 0 8px var(--cyan);flex:none}
.d-title{font-weight:700;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.d-path{font-family:var(--mono);font-size:9.5px;color:var(--tx3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.d-link{margin-left:auto;flex:none;font-family:var(--mono);font-size:10px;letter-spacing:.08em;color:var(--cyan);
  border:1px solid rgba(34,211,238,.35);border-radius:5px;padding:4px 9px;transition:background .15s}
.d-link:hover{background:rgba(34,211,238,.12)}
.d-close{flex:none;background:none;border:1px solid var(--line);color:var(--tx2);border-radius:5px;width:26px;height:26px;cursor:pointer;font-size:13px;line-height:1}
.d-close:hover{color:var(--red);border-color:var(--red)}
.d-body{flex:1;overflow:auto;padding:16px 18px}
.d-body pre{font-family:var(--mono);font-size:11px;line-height:1.75;color:#c6d0e2;white-space:pre-wrap;word-break:break-all}
.d-body pre .yc{color:#4d5a75}          /* 注释 */
.d-body pre .yk{color:var(--cyan)}      /* 键 */
.d-body pre .ys{color:var(--mint)}      /* 字符串 */
.clickable{cursor:pointer}
td.model .m-link{color:var(--tx);border-bottom:1px dashed rgba(34,211,238,.4);cursor:pointer}
td.model .m-link:hover{color:var(--cyan)}
td.yaml .y-link{color:var(--tx3);cursor:pointer}
td.yaml .y-link:hover{color:var(--cyan)}
.bar-row{cursor:pointer}

@media (max-width:1100px){.kpis{grid-template-columns:repeat(3,1fr)}.grid3{grid-template-columns:1fr}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
</head>
<body>
<header>
  <div class="lights"><i></i><i></i><i></i></div>
  <h1>msModelSlim 最佳实践回退率看板 <span>/ lab_practice</span></h1>
  <div class="live"><i></i>MONITOR<b id="h-date"></b></div>
</header>
<main>
  <section class="kpis" id="kpis"></section>

  <section class="grid3">
    <div class="panel">
      <div class="phead"><h2>回退计算量占比 TOP 15</h2><span class="sub">FP 回退 + 升位宽回退 · 点击条形查看对应 YAML</span>
        <span class="legend" style="margin-left:auto"><span><i style="background:var(--red)"></i>FP</span><span><i style="background:var(--amber)"></i>升位宽</span></span></div>
      <div class="pbody"><svg id="ch-bar" width="100%"></svg></div>
    </div>
    <div class="panel">
      <div class="phead"><h2>目标位宽 vs 等效位宽</h2><span class="sub">权重侧</span></div>
      <div class="pbody"><svg id="ch-scatter" width="100%"></svg></div>
    </div>
    <div class="panel">
      <div class="phead"><h2>等效权重位宽分布</h2><span class="sub">可核算配置行</span></div>
      <div class="pbody"><svg id="ch-hist" width="100%"></svg></div>
    </div>
  </section>

  <section class="panel">
    <div class="phead"><h2>逐配置明细</h2><span class="sub">点击表头排序 · 点击模型/YAML 查看量化配置原文 · 悬停聚焦行</span></div>
    <div class="ttools">
      <input id="q" type="text" placeholder="过滤：模型 / YAML / 备注…">
      <label class="grp"><input type="checkbox" id="grp"> 按模型分组</label>
      <span class="cnt" id="cnt"></span>
    </div>
    <div class="twrap"><table id="tbl"><thead></thead><tbody></tbody></table></div>
  </section>
</main>
<div class="tip" id="tip"></div>
<div class="backdrop" id="backdrop"></div>
<aside class="drawer" id="drawer">
  <div class="d-head">
    <span class="d-dot"></span>
    <div style="min-width:0"><div class="d-title" id="d-title"></div><div class="d-path" id="d-path"></div></div>
    <a class="d-link" id="d-link" target="_blank" rel="noopener">GITCODE ↗</a>
    <button class="d-close" id="d-close" title="关闭 (Esc)">✕</button>
  </div>
  <div class="d-body"><pre id="d-pre"></pre></div>
</aside>
<footer>SOURCE: gitcode.com/Ascend/msmodelslim · lab_practice · 口径：回退模块参数 ÷ 全模型线性层参数 · FP 按 16bit 计</footer>

<script>
const DATA = "__DATA__";
const S = DATA.summary, ROWS = DATA.rows;
document.getElementById('h-date').textContent = S.yamls + ' YAMLS / ' + S.rows + ' ROWS';

/* ---------- KPI ---------- */
const kpis = [
  ['YAML 配置', S.yamls, '', 'c-cyan', '最佳实践配置文件'],
  ['配置×模型记录', S.rows, '', 'c-cyan', '一个 YAML 多模型展开'],
  ['FP 回退均值', S.fp_avg, '%', 'c-red', '最大 ' + S.fp_max + '%（不含目标即FP16行）'],
  ['升位宽回退均值', S.up_avg, '%', 'c-amber', '最大 ' + S.up_max + '%'],
  ['等效位宽均值', S.eqw_avg, 'bit', 'c-mint', '范围 ' + S.eqw_min + '~' + S.eqw_max],
  ['FA3 / KVC 启用', S.fa3_en + ' / ' + S.kvc_en, '', 'c-magenta', '配置行数'],
];
document.getElementById('kpis').innerHTML = kpis.map(k =>
  `<div class="kpi ${k[3]}"><div class="k-label">${k[0]}</div><div class="k-val">${k[1]}<span class="k-unit">${k[2]}</span></div><div class="k-sub">${k[4]}</div></div>`).join('');

/* ---------- YAML 抽屉 ---------- */
const YAMLS = DATA.yamls || {};
const drawer = document.getElementById('drawer'), backdrop = document.getElementById('backdrop');
function hlYaml(src){
  const esc = src.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  return esc.split('\n').map(line => {
    const ci = line.indexOf('#');
    if (ci === 0 || (ci > 0 && /^\s*$/.test(line.slice(0,ci))))
      return '<span class="yc">' + line + '</span>';
    // 行内注释（ preceded by 空白 ）单独着色
    let hashAt = -1;
    const m2 = line.match(/\s+#/);
    if (m2) hashAt = line.indexOf(' #');
    let body = line, tail = '';
    if (hashAt > 0) { body = line.slice(0, hashAt); tail = '<span class="yc">' + line.slice(hashAt) + '</span>'; }
    const m = body.match(/^(\s*-?\s*)([\w.*{}$]+)(:)/);
    if (m) body = m[1] + '<span class="yk">' + m[2] + '</span>' + m[3] + body.slice(m[0].length);
    return body + tail;
  }).join('\n');
}
function openYaml(yamlPath, model){
  const txt = YAMLS[yamlPath];
  document.getElementById('d-title').textContent = model + ' — 量化配置';
  document.getElementById('d-path').textContent = 'lab_practice/' + yamlPath;
  document.getElementById('d-link').href = 'https://gitcode.com/Ascend/msmodelslim/blob/master/lab_practice/' + yamlPath;
  document.getElementById('d-pre').innerHTML = txt ? hlYaml(txt) : '（未找到 YAML 原文）';
  drawer.classList.add('show'); backdrop.classList.add('show');
  hideTip();
}
function closeYaml(){ drawer.classList.remove('show'); backdrop.classList.remove('show'); }
document.getElementById('d-close').addEventListener('click', closeYaml);
backdrop.addEventListener('click', closeYaml);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeYaml(); });

/* ---------- utils ---------- */
const tip = document.getElementById('tip');
function showTip(html, x, y){ tip.innerHTML = html; tip.style.display='block';
  tip.style.left = Math.min(x+14, innerWidth-380)+'px'; tip.style.top = (y+14)+'px'; }
function hideTip(){ tip.style.display='none'; }
const NS = 'http://www.w3.org/2000/svg';
function el(t, a){ const e=document.createElementNS(NS,t); for(const k in a) e.setAttribute(k,a[k]); return e; }

/* ---------- TOP15 横向条形 ---------- */
(function(){
  const rows = ROWS.filter(r => r.fp_share != null && (r.fp_share + (r.up_share||0)) > 0.01)
                   .sort((a,b) => (b.fp_share+(b.up_share||0)) - (a.fp_share+(a.up_share||0))).slice(0,15);
  const svg = document.getElementById('ch-bar');
  const W = 560, rh = 30, pad = {l:190, r:46, t:6, b:6}, H = rows.length*rh + pad.t + pad.b;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const max = Math.max(...rows.map(r => r.fp_share + (r.up_share||0)), 1);
  rows.forEach((r,i) => {
    const y = pad.t + i*rh, bw = W - pad.l - pad.r;
    const g = el('g',{class:'bar-row'});
    g.appendChild(el('rect',{class:'bar-bg', x:pad.l, y:y+4, width:bw, height:rh-8, rx:3, fill:'transparent'}));
    const t1 = el('text',{x:pad.l-8, y:y+rh/2+3.5, 'text-anchor':'end', 'font-size':10, fill:'#e5eaf3'});
    t1.textContent = r.model.length>16 ? r.model.slice(0,15)+'…' : r.model;
    const t2 = el('text',{x:pad.l-8, y:y+rh/2+13, 'text-anchor':'end', 'font-size':8, fill:'#566179'});
    t2.textContent = r.mode;
    g.appendChild(t1); g.appendChild(t2);
    const w1 = r.fp_share/max*bw, w2 = (r.up_share||0)/max*bw;
    if (w1>0) g.appendChild(el('rect',{x:pad.l, y:y+5, width:w1, height:rh-10, rx:2, fill:'#f87171'}));
    if (w2>0) g.appendChild(el('rect',{x:pad.l+w1, y:y+5, width:w2, height:rh-10, rx:2, fill:'#fbbf24'}));
    const tv = el('text',{x:pad.l+w1+w2+6, y:y+rh/2+3.5, 'font-size':10, fill:'#8b96ad'});
    tv.textContent = (r.fp_share+(r.up_share||0)).toFixed(1)+'%';
    g.appendChild(tv);
    g.addEventListener('mousemove', e => showTip(
      `<b>${r.model}</b> <span class="t-m">${r.yaml}</span><br>FP 回退 <span style="color:#f87171">${r.fp_share}%</span> · 升位宽 <span style="color:#fbbf24">${r.up_share||0}%</span><br>attn: ${r.attn_share||'—'}<br>mlp: ${r.mlp_share||'—'}`, e.clientX, e.clientY));
    g.addEventListener('mouseleave', hideTip);
    g.addEventListener('click', () => openYaml(r.yaml, r.model));
    svg.appendChild(g);
  });
})();

/* ---------- 目标位宽 vs 等效位宽散点 ---------- */
(function(){
  const rows = ROWS.filter(r => r.eq_w != null && r.w);
  const svg = document.getElementById('ch-scatter');
  const W = 380, H = 300, pad = {l:38, r:10, t:10, b:30};
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const x0=3, x1=17, y0=3, y1=17;
  const X = v => pad.l + (v-x0)/(x1-x0)*(W-pad.l-pad.r);
  const Y = v => H-pad.b - (v-y0)/(y1-y0)*(H-pad.t-pad.b);
  [4,8,16].forEach(v => {
    svg.appendChild(el('line',{x1:X(v),y1:pad.t,x2:X(v),y2:H-pad.b,stroke:'#1a2540','stroke-width':1}));
    const t=el('text',{x:X(v),y:H-pad.b+14,'text-anchor':'middle','font-size':9,fill:'#566179'}); t.textContent='w'+v; svg.appendChild(t);
    svg.appendChild(el('line',{x1:pad.l,y1:Y(v),x2:W-pad.r,y2:Y(v),stroke:'#1a2540','stroke-width':1}));
    const t2=el('text',{x:pad.l-6,y:Y(v)+3,'text-anchor':'end','font-size':9,fill:'#566179'}); t2.textContent=v; svg.appendChild(t2);
  });
  svg.appendChild(el('line',{x1:X(4),y1:Y(4),x2:X(16),y2:Y(16),stroke:'#22d3ee','stroke-width':1,'stroke-dasharray':'4 4',opacity:.5}));
  const td=el('text',{x:X(13.2),y:Y(14.2),'font-size':8,fill:'#22d3ee',opacity:.7}); td.textContent='无回退线'; svg.appendChild(td);
  const col = f => f==null ? '#566179' : f>=20 ? '#f87171' : f>=5 ? '#fbbf24' : '#34d399';
  rows.forEach(r => {
    const c = el('circle',{cx:X(r.w+(Math.random()-.5)*.5), cy:Y(Math.min(r.eq_w,16.5)+(Math.random()-.5)*.3),
      r: 2.5+Math.sqrt(Math.min(r.lin_total||1,3000))/22*6, fill:col(r.fp_share), 'fill-opacity':.55, stroke:col(r.fp_share),'stroke-width':1});
    c.addEventListener('mousemove', e => showTip(
      `<b>${r.model}</b> <span class="t-m">${r.mode}</span><br>目标 w${r.w} → 等效 ${r.eq_w} bit<br>FP 回退 ${r.fp_share==null?(r.lin_total==null?'N/A':'—（目标即FP16）'):r.fp_share+'%'}`,
      e.clientX, e.clientY));
    c.addEventListener('mouseleave', hideTip);
    svg.appendChild(c);
  });
})();

/* ---------- 等效位宽直方图 ---------- */
(function(){
  const vals = ROWS.filter(r => r.eq_w != null).map(r => r.eq_w);
  const bins = {};
  vals.forEach(v => { const b = Math.round(v*2)/2; bins[b]=(bins[b]||0)+1; });
  const keys = Object.keys(bins).map(Number).sort((a,b)=>a-b);
  const svg = document.getElementById('ch-hist');
  const W = 380, H = 300, pad = {l:30, r:10, t:14, b:30};
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const max = Math.max(...Object.values(bins));
  const bw = (W-pad.l-pad.r)/keys.length;
  keys.forEach((k,i) => {
    const h = bins[k]/max*(H-pad.t-pad.b);
    const x = pad.l+i*bw, y = H-pad.b-h;
    const c = k<=4.5 ? '#34d399' : k<=8.5 ? '#22d3ee' : '#fbbf24';
    svg.appendChild(el('rect',{x:x+1.5, y, width:bw-3, height:h, rx:2, fill:c, 'fill-opacity':.8}));
    const t=el('text',{x:x+bw/2, y:y-4,'text-anchor':'middle','font-size':9,fill:'#8b96ad'}); t.textContent=bins[k]; svg.appendChild(t);
    const tl=el('text',{x:x+bw/2, y:H-pad.b+14,'text-anchor':'middle','font-size':8.5,fill:'#566179'}); tl.textContent=k; svg.appendChild(tl);
  });
  const cap=el('text',{x:pad.l,y:10,'font-size':8.5,fill:'#566179'}); cap.textContent='横轴：等效位宽 bit（0.5 桶）· 纵轴：配置行数'; svg.appendChild(cap);
})();

/* ---------- 明细表 ---------- */
const COLS = [
  ['seq','#','num'], ['model','模型','s'], ['yaml','YAML','s'], ['mode','模式','s'],
  ['attn_scheme','attn方案','s'], ['attn_share','attn回退','s'],
  ['mlp_scheme','mlp方案','s'], ['mlp_share','mlp回退','s'],
  ['fp_share','FP%','num'], ['up_share','升位宽%','num'],
  ['eq_w','等效W','num'], ['eq_a','等效A','num'], ['note','备注','s'],
];
let sortKey='fp_share', sortDir=-1, query='', grpMode=false;
const thead = document.querySelector('#tbl thead'), tbody = document.querySelector('#tbl tbody');
thead.innerHTML = '<tr>' + COLS.map(c => `<th data-k="${c[0]}" data-t="${c[2]}">${c[1]}<span class="arr"></span></th>`).join('') + '</tr>';
function colorize(cell){
  if (!cell) return '<span style="color:#3a4459">—</span>';
  return cell.replace(/(FP:[^；<]+)/g,'<span class="fp">$1</span>')
             .replace(/(升位宽[^；<]+)/g,'<span class="up">$1</span>')
             .replace(/(不量化（FP16）)/g,'<span class="fp">$1</span>');
}
function rowHtml(r, i){
  return `<tr style="animation-delay:${Math.min(i*18,900)}ms">
    <td class="num">${r.seq}</td><td class="model"><span class="m-link" data-y="${r.yaml}" data-m="${r.model}">${r.model}</span></td><td class="yaml"><span class="y-link" data-y="${r.yaml}" data-m="${r.model}" title="${r.yaml}">${r.yaml}</span></td>
    <td><span class="badge">${r.mode}</span></td>
    <td>${r.attn_scheme||'—'}</td><td>${colorize(r.attn_share)}</td>
    <td>${r.mlp_scheme||'—'}</td><td>${colorize(r.mlp_share)}</td>
    <td class="num">${r.fp_share==null?(r.lin_total==null?'N/A':'—'):r.fp_share}</td><td class="num">${r.up_share==null?'—':r.up_share}</td>
    <td class="num">${r.eq_w??'N/A'}</td><td class="num">${r.eq_a??'N/A'}</td>
    <td class="note">${r.note||''}</td></tr>`;
}
function render(){
  let rows = ROWS.filter(r => !query || (r.model+r.yaml+(r.note||'')).toLowerCase().includes(query));
  const t = COLS.find(c=>c[0]===sortKey)[2];
  const cmp = (a,b) => {
    let x=a[sortKey], y=b[sortKey];
    if (x==null) return 1; if (y==null) return -1;
    return (t==='num' ? x-y : String(x).localeCompare(String(y),'zh')) * sortDir;
  };
  rows.sort(cmp);
  if (grpMode){
    const groups = {};
    rows.forEach(r => { (groups[r.family||r.model] = groups[r.family||r.model]||[]).push(r); });
    const gkeys = Object.keys(groups).sort((g1,g2) => {
      const s = g => Math.max(...groups[g].map(r => (r.fp_share||0)+(r.up_share||0)));
      return s(g2)-s(g1);
    });
    let html = '', i = 0;
    for (const g of gkeys){
      const gr = groups[g];
      const fps = gr.map(r=>r.fp_share).filter(v=>v!=null);
      const fpTxt = fps.length ? (Math.min(...fps)===Math.max(...fps) ? 'FP '+Math.min(...fps)+'%' : 'FP '+Math.min(...fps)+'~'+Math.max(...fps)+'%') : 'FP —';
      const models = [...new Set(gr.map(r=>r.model))];
      html += `<tr class="g-head"><td colspan="13">${g}<span class="g-n">${gr.length} 行 · ${models.length} 模型</span><span class="g-r">${fpTxt}</span></td></tr>`;
      html += gr.map(r => rowHtml(r, i++)).join('');
    }
    tbody.innerHTML = html;
  } else {
    tbody.innerHTML = rows.map((r,i) => rowHtml(r,i)).join('');
  }
  document.getElementById('cnt').textContent = rows.length + ' / ' + ROWS.length + ' ROWS';
  thead.querySelectorAll('th').forEach(th => {
    th.querySelector('.arr').textContent = th.dataset.k===sortKey ? (sortDir>0?'▲':'▼') : '';
  });
}
thead.addEventListener('click', e => {
  const th = e.target.closest('th'); if (!th) return;
  const k = th.dataset.k;
  if (sortKey===k) sortDir*=-1; else { sortKey=k; sortDir = (th.dataset.t==='num'?-1:1); }
  render();
});
tbody.addEventListener('click', e => {
  const l = e.target.closest('.m-link,.y-link'); if (!l) return;
  openYaml(l.dataset.y, l.dataset.m);
});
document.getElementById('q').addEventListener('input', e => { query = e.target.value.trim().toLowerCase(); render(); });
document.getElementById('grp').addEventListener('change', e => { grpMode = e.target.checked; render(); });
render();
</script>
</body>
</html>
"""

payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
html = TEMPLATE.replace('"__DATA__"', payload)
out_path = os.path.join(OUT, '最佳实践回退率看板.html')
with open(out_path, 'w') as f2:
    f2.write(html)
print('dashboard saved:', out_path)
