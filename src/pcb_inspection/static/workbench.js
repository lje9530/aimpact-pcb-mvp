'use strict';

const $ = selector => document.querySelector(selector);
const escape = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[ch]));
const route = id => `/api/inspections/${id}`;
const labels = {OK: '정상', NG: '불량', WAIT: '판정 보류'};
const pages = {
  workbench: ['검사 작업대', 'PCB 검사 영상에서 결함 위치와 유형을 검토하고 최종판정을 기록합니다.'],
  dashboard: ['품질 현황', '최종판정을 기준으로 유형별 발생 건수와 반복되는 불량을 확인합니다.'],
  history: ['검사 이력', '제품, 기간, 불량유형으로 검색하고 조회 결과를 엑셀로 내려받습니다.'],
};
const state = {items: [], classes: {}, item: null, selected: null, queue: 'all', view: 'workbench', mode: 'inspect', focus: null, zoom: 1, cutoff: .7, showBoxes: true};
try {
  const saved = Number(localStorage.getItem('aimpact-cutoff'));
  if (saved >= .3 && saved <= .95) state.cutoff = saved;
} catch (_) {}
let toastTimer;

async function api(url, options) {
  const resp = await fetch(url, options);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `요청에 실패했습니다. (${resp.status})`);
  return data;
}

function toast(message, danger = false) {
  clearTimeout(toastTimer);
  const element = $('#toast');
  element.textContent = message;
  element.className = `toast ${danger ? 'danger' : ''}`;
  element.hidden = false;
  toastTimer = setTimeout(() => { element.hidden = true; }, 4000);
}

function badge(result, text = labels[result] || '검토 대기') {
  return `<span class="pill ${result === 'OK' ? 'ok' : result === 'NG' ? 'ng' : 'wait'}">${text}</span>`;
}

const percent = value => value === null ? '—' : `${Math.round(value * 100)}%`;
const displayDate = value => escape(value ? value.replace('T', ' ').slice(0, 16) : '—');
const typeNames = values => (values ?? []).map(kind => escape(state.classes[kind])).join(', ') || '—';
const needsReview = item => !item.final_result && (item.ai_result === 'WAIT' || item.score === null || item.score < state.cutoff);
const isDetector = item => item.model_version.startsWith('pcb-yolo');

function summary() {
  const done = state.items.filter(item => item.final_result);
  const comparable = done.filter(item => item.ai_result !== 'WAIT');
  const pending = state.items.length - done.length;
  const agree = comparable.filter(item => item.final_result === item.ai_result).length;
  const metrics = [
    ['전체 검사', state.items.length, '건', `${new Set(state.items.map(item => item.supplier)).size}개 협력사`, 'neutral'],
    ['검토 대기', pending, '건', `확인 필수 ${state.items.filter(needsReview).length}건`, 'attention'],
    ['최종판정 완료', done.length, '건', `불량 확정 ${done.filter(item => item.final_result === 'NG').length}건`, 'neutral'],
    ['AI·담당자 일치', comparable.length ? percent(agree / comparable.length) : '—', '', `정상·불량 일치 ${agree}/${comparable.length}건`, 'match'],
  ];
  $('#summary').innerHTML = metrics.map(([name, value, unit, hint, tone]) => `<article class="metric ${tone}"><span>${name}</span><strong>${value}<small>${unit}</small></strong><div>${hint}</div></article>`).join('');
}

function queueItems() {
  if (state.queue === 'pending') return state.items.filter(item => !item.final_result);
  if (state.queue === 'done') return state.items.filter(item => item.final_result);
  return state.items;
}

function queue() {
  const rows = queueItems();
  $('#queue-count').textContent = `${rows.length}건`;
  $('#queue-filters').innerHTML = [['all', '전체'], ['pending', '대기'], ['done', '완료']].map(([value, name]) => `<button type="button" data-queue="${value}" aria-pressed="${state.queue === value}">${name}</button>`).join('');
  $('#queue-items').innerHTML = rows.map(item => {
    const caption = item.final_result ? '완료' : needsReview(item) ? '확인 필수' : labels[item.ai_result];
    const kind = item.final_result || (needsReview(item) ? 'WAIT' : item.ai_result);
    return `<button class="queue-row" type="button" data-id="${item.id}" aria-pressed="${state.selected === item.id}">
      <div class="thumb"><img loading="lazy" alt="" src="${route(item.id)}/image/test"></div><span class="queue-text"><strong>${escape(item.product)}</strong><small>${escape(item.external_id)}</small><small>${escape(item.supplier)}</small></span>${badge(kind, caption)}</button>`;
  }).join('') || '<p class="empty">표시할 검사 기록이 없습니다.</p>';
}

function boxStyle(item, [x1, y1, x2, y2]) {
  return `left:${100*x1/item.width}%;top:${100*y1/item.height}%;width:${100*(x2-x1)/item.width}%;height:${100*(y2-y1)/item.height}%`;
}

function plate(item, variant, overlay = false) {
  const region = Number.isInteger(state.focus) ? item.signals[state.focus]?.box : null;
  const anchor = region ? [(region[0] + region[2]) / 2 / item.width, (region[1] + region[3]) / 2 / item.height] : [.5, .5];
  const transform = `transform:scale(${state.zoom});transform-origin:${anchor[0] * 100}% ${anchor[1] * 100}%`;
  const showDetection = variant === 'test' && !overlay && isDetector(item) && state.showBoxes;
  const detections = showDetection ? item.signals.slice(0, 20).map((signal, index) => {
    if (region && state.focus !== index) return '';
    const label = `${state.classes[signal.type]} ${percent(signal.score)}`;
    const y = signal.box[1] / item.height;
    const x = signal.box[0] / item.width;
    return `<button type="button" class="detection-box${y < .08 ? ' inset-label' : ''}${x > .78 ? ' right-label' : ''}" data-signal="${index}" style="${boxStyle(item, signal.box)}" aria-label="${escape(label)} 확대">${state.zoom === 1 ? `<span>${escape(label)}</span>` : ''}</button>`;
  }).join('') : '';
  const marker = region && !showDetection ? `<i class="focus-mark" style="${boxStyle(item, region)}"></i>` : '';
  const title = variant === 'test' ? '검사 영상' : '기준 영상';
  const source = `${route(item.id)}/image/${variant}`;
  const style = `--ratio:${item.width} / ${item.height};--image-width:${item.width / item.height * 100}cqh`;
  return `<figure class="capture"><div class="plate-holder"><div class="plate" style="${style}"><div class="plate-layer" style="${transform}">
    <img src="${source}" alt="PCB ${title}" draggable="false">
    ${overlay ? `<img class="overlay" src="${route(item.id)}/difference" alt="" draggable="false">` : ''}${detections}${marker}
    </div></div></div><figcaption>${title}${overlay ? ' · 이진 차이' : ''}</figcaption></figure>`;
}

function viewer() {
  const item = state.item;
  if (!item) { $('#viewer').innerHTML = '<div class="empty">검사 대기열에서 영상을 선택하세요.</div>'; return; }
  const paired = state.mode === 'compare';
  const source = item.data_source.startsWith('DeepPCB') ? (item.external_id.endsWith('-CTRL') ? 'DeepPCB · 정상 대조군' : 'DeepPCB · 연구 샘플') : item.data_source;
  const selected = Number.isInteger(state.focus) ? item.signals[state.focus] : null;
  $('#viewer').innerHTML = `
    <div class="panel-header item-header"><div><h2>${escape(item.product)}</h2><div class="muted item-meta">${escape(item.external_id)} <span>·</span> ${displayDate(item.inspected_at)}</div></div><span class="source">${escape(source)}</span></div>
    <div class="viewer-toolbar"><div class="segments" aria-label="영상 보기">${[['inspect','검사 영상'],['compare','나란히 비교'],['difference','차이 보기']].map(([mode, text]) => `<button type="button" data-mode="${mode}" aria-pressed="${state.mode === mode}" ${mode !== 'inspect' && !item.reference_file ? 'disabled' : ''}>${text}</button>`).join('')}</div><div class="viewer-actions">${isDetector(item) ? `<button type="button" class="subtle zoom" data-boxes aria-pressed="${state.showBoxes}" ${state.mode === 'difference' ? 'disabled' : ''}>탐지 박스</button>` : ''}<button type="button" class="subtle zoom" data-zoom aria-label="확대 전환">${state.zoom}×</button></div></div>
    <div class="imaging ${paired ? 'paired' : ''}">${plate(item, 'test', state.mode === 'difference')}${paired ? plate(item, 'reference') : ''}</div>
    <div class="viewer-foot"><span>${item.width} × ${item.height} px${item.roi ? ' · ROI 지정' : ''}</span><span>${selected ? `선택 위치 (${selected.box[0]}, ${selected.box[1]}) px` : '후보를 선택해 확대'}</span></div>
    <div class="signals"><div class="signals-head"><h3>${isDetector(item) ? 'YOLO 검출 결과' : '모델 후보'}</h3><span class="muted">${item.signals.length}곳</span></div>
      ${item.signals.length ? `<div class="signal-list">${item.signals.slice(0, 20).map((signal, index) => `<button type="button" data-signal="${index}" aria-pressed="${state.focus === index}"><span class="target" aria-hidden="true">⌖</span><b>${escape(state.classes[signal.type])}</b><span>${percent(signal.score)}</span></button>`).join('')}</div>${item.signals.length > 20 ? '<p class="muted small">점수 상위 20곳 표시</p>' : ''}` : `<p class="muted">${item.ai_result === 'WAIT' ? '기준 영상이 없어 후보를 계산하지 않았습니다.' : '검출된 후보가 없습니다.'}</p>`}
    </div>`;
}

function review() {
  const item = state.item;
  if (!item) { $('#review').innerHTML = '<div class="empty">선택된 검사 기록이 없습니다.</div>'; return; }
  const scoreHelp = item.ai_result === 'WAIT' ? '정렬된 기준 영상 필요' : item.score === null ? '검출 결과 없음' : `모델 점수 ${percent(item.score)}`;
  const startingResult = item.final_result || (item.ai_result === 'WAIT' ? '' : item.ai_result);
  const chosen = item.final_types || item.ai_types;
  $('#review').innerHTML = `
    <div class="panel-header"><h2>판정 검토</h2>${item.final_result ? badge('OK', '검토 완료') : badge('WAIT', needsReview(item) ? '확인 필수' : '검토 대기')}</div>
    <div class="review-body">
      <div class="ai-card"><div><h3>${isDetector(item) ? 'YOLO 판정 초안' : 'AI 판정 초안'}</h3>${badge(item.ai_result)}</div><strong>${typeNames(item.ai_types)}</strong><div class="muted small">${scoreHelp}${item.ms === null ? '' : ` · CPU ${item.ms.toFixed(1)} ms`}</div></div>
      <dl class="facts"><div><dt>협력사 판정</dt><dd>${escape(labels[item.reported_result] || item.reported_result || '—')}${item.reported_type ? ` · ${escape(item.reported_type)}` : ''}</dd></div><div><dt>협력사 · 로트</dt><dd>${escape(item.supplier)} · ${escape(item.batch || '—')}</dd></div></dl>
      <form id="review-form">
        <h3>담당자 최종판정</h3>
        <div class="decision">${[['OK', '정상'], ['NG','불량']].map(([value, text]) => `<label><input type="radio" name="result" value="${value}" ${startingResult === value ? 'checked' : ''} required><span>${text}</span></label>`).join('')}</div>
        <fieldset id="review-types" ${startingResult !== 'NG' ? 'hidden disabled' : ''}><legend>불량유형</legend><div class="type-choices">${Object.entries(state.classes).filter(([id]) => id !== '0').map(([id, name]) => `<label><input type="checkbox" name="kind" value="${id}" ${chosen.includes(Number(id)) ? 'checked' : ''}><span>${escape(name)}</span></label>`).join('')}</div></fieldset>
        <label>담당자<input name="reviewer" placeholder="담당자 이름" value="${escape(item.reviewer === '데모 담당자' ? '' : item.reviewer || '')}" required maxlength="60"></label>
        <label>검토 메모 <span class="muted">선택</span><textarea name="note" rows="2" maxlength="500" placeholder="영상에서 확인한 근거를 기록하세요">${escape(item.note === '공개 라벨 기반 시연용 검토 기록' ? '' : item.note || '')}</textarea></label>
        <button type="submit" class="primary">최종판정 저장</button>
      </form>
    </div>
    ${item.history.length ? `<details class="audit"><summary>판정 이력 ${item.history.length}건</summary><ul>${item.history.map(entry => `<li><b>${entry.revision}차 · ${labels[entry.result]} · ${escape(entry.reviewer)}</b><br><span>${displayDate(entry.created_at)} UTC</span>${entry.note ? `<p>${escape(entry.note)}</p>` : ''}</li>`).join('')}</ul></details>` : ''}`;
}

async function selectItem(id) {
  state.selected = id;
  queue();
  if (!id) { state.item = null; viewer(); review(); return; }
  try {
    const item = await api(route(id));
    if (state.selected !== id) return;
    state.item = item;
    state.mode = item.reference_file && !isDetector(item) ? 'difference' : 'inspect';
    state.focus = null;
    state.zoom = 1;
    state.showBoxes = true;
    viewer(); review();
  } catch (error) { toast(error.message, true); }
}

async function loadItems() {
  const data = await api('/api/inspections');
  state.items = data.items;
  state.classes = data.classes;
  summary(); queue();
  if (state.view === 'dashboard') dashboard();
  if (state.view === 'history') history();
}

function dashboard() {
  const done = state.items.filter(item => item.final_result);
  const confirmedNg = done.filter(item => item.final_result === 'NG');
  const types = Object.entries(state.classes).filter(([type]) => type !== '0').map(([type, name]) => [name, confirmedNg.filter(item => item.final_types.includes(Number(type))).length]);
  const max = Math.max(1, ...types.map(([, count]) => count));
  const byDay = [...new Set(state.items.map(item => item.inspected_at.slice(0, 10)))].sort();
  const repeats = new Map();
  confirmedNg.forEach(item => item.final_types.forEach(type => {
    const key = `${item.product} · ${state.classes[type]}`;
    repeats.set(key, (repeats.get(key) || 0) + 1);
  }));
  $('#dashboard').innerHTML = `
    <div class="dashboard-grid">
      <section class="panel chart-box"><div class="panel-header"><h2>확정 불량유형</h2><span class="muted">복수 유형 집계</span></div><div class="chart">${types.map(([name, count]) => `<div><span>${escape(name)}</span><div class="track"><i style="width:${count * 100 / max}%"></i></div><b>${count}</b></div>`).join('')}</div></section>
      <section class="panel chart-box"><div class="panel-header"><h2>일자별 검사 현황</h2><span class="muted">검사일 기준</span></div><div class="trend-wrap"><table><thead><tr><th>검사일</th><th>전체</th><th>검토 완료</th><th>불량 확정</th></tr></thead><tbody>${byDay.map(day => { const items = state.items.filter(item => item.inspected_at.slice(0, 10) === day); return `<tr><td>${escape(day)}</td><td>${items.length}</td><td>${items.filter(item => item.final_result).length}</td><td>${items.filter(item => item.final_result === 'NG').length}</td></tr>`; }).join('') || '<tr><td colspan="4">검사 기록 없음</td></tr>'}</tbody></table></div></section>
      <section class="panel chart-box"><div class="panel-header"><h2>반복 불량</h2><span class="muted">같은 제품·유형 2건 이상</span></div><ul class="repeat-list">${[...repeats.entries()].filter(([, count]) => count >= 2).sort((a, b) => b[1] - a[1]).map(([name, count]) => `<li>${escape(name)}<b>${count}건</b></li>`).join('') || '<li class="muted">반복 확정된 유형이 없습니다.</li>'}</ul></section>
      <section class="panel chart-box"><div class="panel-header"><h2>담당자 확인 기준</h2></div><div class="cutoff"><label for="cutoff">낮은 점수 기준 <output id="cutoff-value">${percent(state.cutoff)}</output></label><input id="cutoff" type="range" min="30" max="95" value="${Math.round(state.cutoff * 100)}"><p class="muted">점수가 이 값보다 낮거나 점수가 없는 미검토 건에 확인 필수를 표시합니다.</p></div></section>
    </div>`;
}

function history() {
  const products = [...new Set(state.items.map(item => item.product))].sort();
  $('#history').innerHTML = `<div class="panel history-panel"><div class="panel-header"><h2>검사 기록 조회</h2><a id="export" class="secondary" href="/api/export.xlsx" download>엑셀 내려받기</a></div>
    <div class="history-filters"><label>제품<select id="filter-product"><option value="">전체 제품</option>${products.map(product => `<option>${escape(product)}</option>`).join('')}</select></label>
    <label>시작일<input id="filter-start" type="date"></label><label>종료일<input id="filter-end" type="date"></label>
    <label>최종 불량유형<select id="filter-type"><option value="">전체 유형</option>${Object.entries(state.classes).filter(([id]) => id !== '0').map(([id, name]) => `<option value="${id}">${escape(name)}</option>`).join('')}</select></label></div>
    <p class="table-count muted" id="history-count"></p><div class="table-scroll"><table><thead><tr><th>검사 ID · 제품</th><th>협력사</th><th>검사일시</th><th>AI 초안</th><th>최종판정</th><th>확정 불량유형</th><th>담당자</th></tr></thead><tbody id="history-rows"></tbody></table></div></div>`;
  historyRows();
}

function historyRows() {
  const product = $('#filter-product').value, kind = $('#filter-type').value;
  const start = $('#filter-start').value, end = $('#filter-end').value;
  const matches = state.items.filter(item => {
    const date = item.inspected_at.slice(0, 10);
    return (!product || product === item.product) && (!start || start <= date) && (!end || end >= date) && (!kind || item.final_types?.includes(Number(kind)));
  });
  $('#history-count').textContent = `조회 결과 ${matches.length}건`;
  $('#export').href = matches.length === state.items.length ? '/api/export.xlsx' : `/api/export.xlsx?ids=${matches.map(item => item.id).join(',')}`;
  $('#history-rows').innerHTML = matches.map(item => `<tr><td><button class="record-link" data-show="${item.id}">${escape(item.external_id)}</button><small>${escape(item.product)}</small></td><td>${escape(item.supplier)}</td><td>${displayDate(item.inspected_at)}</td><td>${badge(item.ai_result)}</td><td>${badge(item.final_result)}</td><td>${typeNames(item.final_types)}</td><td>${escape(item.reviewer || '—')}</td></tr>`).join('') || '<tr><td colspan="7">조건에 맞는 검사 기록이 없습니다.</td></tr>';
}

function showPage() {
  state.view = Object.hasOwn(pages, location.hash.slice(1)) ? location.hash.slice(1) : 'workbench';
  const [title, description] = pages[state.view];
  $('#page-title').textContent = title;
  $('#page-description').textContent = description;
  for (const key of Object.keys(pages)) {
    document.getElementById(key).hidden = key !== state.view;
    const link = document.querySelector(`[data-view="${key}"]`);
    if (key === state.view) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  if (state.view === 'dashboard') dashboard();
  if (state.view === 'history') history();
}

$('#queue-filters').addEventListener('click', event => {
  const value = event.target.closest('[data-queue]')?.dataset.queue;
  if (!value) return;
  state.queue = value; queue();
  const rows = queueItems();
  if (!rows.some(item => item.id === state.selected)) selectItem(rows[0]?.id);
});
$('#queue-items').addEventListener('click', event => {
  const button = event.target.closest('[data-id]');
  if (button) selectItem(Number(button.dataset.id));
});
$('#viewer').addEventListener('click', event => {
  const button = event.target.closest('button');
  if (!button || !state.item) return;
  if (button.dataset.mode) { state.mode = button.dataset.mode; state.focus = null; state.zoom = 1; }
  if ('boxes' in button.dataset) { state.showBoxes = !state.showBoxes; state.focus = null; state.zoom = 1; }
  if ('zoom' in button.dataset) { if (state.zoom === 3) state.focus = null; state.zoom = state.zoom === 1 ? 2 : state.zoom === 2 ? 3 : 1; }
  if ('signal' in button.dataset) { state.focus = Number(button.dataset.signal); state.zoom = 3; }
  viewer();
});
$('#review').addEventListener('change', event => {
  if (event.target.name !== 'result') return;
  const show = event.target.value === 'NG';
  $('#review-types').hidden = !show;
  $('#review-types').disabled = !show;
});
$('#review').addEventListener('submit', async event => {
  if (event.target.id !== 'review-form' || !state.item) return;
  event.preventDefault();
  const data = new FormData(event.target), submit = event.submitter;
  const itemId = state.item.id;
  submit.disabled = true;
  try {
    const updated = await api(`${route(itemId)}/review`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
      result: data.get('result'), types: data.getAll('kind').map(Number), reviewer: data.get('reviewer'), note: data.get('note'), revision: state.item.revision || 0,
    })});
    if (state.selected === itemId) { state.item = updated; review(); }
    await loadItems();
    if (state.queue === 'pending' && state.selected === itemId) await selectItem(queueItems()[0]?.id);
    toast('최종판정이 저장되었습니다.');
  } catch (error) { toast(error.message, true); }
  finally { submit.disabled = false; }
});
$('#dashboard').addEventListener('input', event => {
  if (event.target.id !== 'cutoff') return;
  state.cutoff = Number(event.target.value) / 100;
  $('#cutoff-value').textContent = percent(state.cutoff);
  try { localStorage.setItem('aimpact-cutoff', String(state.cutoff)); } catch (_) {}
  summary(); queue(); review();
});
$('#history').addEventListener('input', event => { if (event.target.id.startsWith('filter-')) historyRows(); });
$('#history').addEventListener('click', async event => {
  const target = event.target.closest('[data-show]');
  if (target) { state.queue = 'all'; location.hash = 'workbench'; await selectItem(Number(target.dataset.show)); }
});
window.addEventListener('hashchange', showPage);

const quickFile = $('#quick-image-file');
const quickButton = $('#quick-image');
if (quickFile && quickButton) {
  quickButton.addEventListener('click', () => quickFile.click());
  quickFile.addEventListener('change', async () => {
    const file = quickFile.files[0];
    if (!file) return;
    const data = new FormData();
    data.set('image', file);
    quickButton.disabled = true;
    quickButton.textContent = '검사 중…';
    try {
      const result = await api('/api/inspect-image', {method: 'POST', body: data});
      state.queue = 'all'; location.hash = 'workbench';
      await loadItems();
      await selectItem(result.id);
      toast('PCB 검사 결과가 저장되었습니다.');
    } catch (error) { toast(error.message, true); }
    finally {
      quickFile.value = '';
      quickButton.disabled = false;
      quickButton.textContent = 'PCB 이미지 검사';
    }
  });
}

const importForm = $('#import-form'), dialog = $('#import-dialog');
const message = $('#import-message'), mapping = $('#mapping'), extra = $('#import-extra'), submitImport = $('#submit-import');
function resetPreview() {
  mapping.hidden = true;
  extra.hidden = true;
  submitImport.disabled = true;
  message.textContent = '';
}
$('#open-import').addEventListener('click', () => dialog.showModal());
$('#close-import').addEventListener('click', () => dialog.close());
importForm.addEventListener('change', event => { if (['metadata', 'supplier'].includes(event.target.name)) resetPreview(); });
$('#preview-import').addEventListener('click', async event => {
  const control = event.currentTarget;
  const original = new FormData(importForm);
  const file = importForm.elements.metadata.files[0];
  if (!file?.name) { message.textContent = '먼저 CSV 또는 XLSX 파일을 선택하세요.'; return; }
  control.disabled = true;
  resetPreview();
  try {
    const data = new FormData();
    data.set('metadata', file); data.set('supplier', original.get('supplier'));
    const preview = await api('/api/import/preview', {method: 'POST', body: data});
    if (importForm.elements.metadata.files[0] !== file || importForm.elements.supplier.value !== original.get('supplier')) return;
    mapping.innerHTML = `<h3>${preview.count}행 · ${preview.template ? '저장된 매핑 불러옴' : '공통 항목 매핑'}</h3><div class="mapping-grid">${Object.entries(preview.fields).map(([name, label]) => `<label>${escape(label)}${preview.required.includes(name) ? ' *' : ''}<select data-map="${name}" ${preview.required.includes(name) ? 'required' : ''}><option value="">미지정</option>${preview.headers.map(header => `<option value="${escape(header)}" ${preview.mapping[name] === header ? 'selected' : ''}>${escape(header)}</option>`).join('')}</select><small>${escape(preview.first[preview.mapping[name]] || '')}</small></label>`).join('')}</div>`;
    mapping.hidden = false; extra.hidden = false; submitImport.disabled = false;
    mapping.onchange = event => { if (event.target.matches('select')) event.target.parentElement.querySelector('small').textContent = preview.first[event.target.value] || ''; };
  } catch (error) { message.textContent = error.message; }
  finally { control.disabled = false; }
});
importForm.addEventListener('submit', async event => {
  event.preventDefault();
  const data = new FormData(importForm);
  const columns = Object.fromEntries([...mapping.querySelectorAll('[data-map]')].map(input => [input.dataset.map, input.value]));
  data.set('mapping', JSON.stringify(columns));
  const roi = String(data.get('roi')).trim();
  data.set('roi', roi ? JSON.stringify(roi.split(',').map(value => value.trim() ? Number(value.trim()) : null)) : 'null');
  submitImport.disabled = true;
  $('#preview-import').disabled = true;
  message.textContent = '영상을 분석하고 검사 기록을 저장하고 있습니다.';
  try {
    const result = await api('/api/import', {method: 'POST', body: data});
    dialog.close(); importForm.reset(); resetPreview();
    state.queue = 'all'; location.hash = 'workbench';
    await loadItems();
    await selectItem(result.ids[0] || state.selected || state.items[0]?.id);
    toast(`${result.imported}건 가져오기 완료${result.duplicates ? ` · 중복 ${result.duplicates}건 건너뜀` : ''}`);
  } catch (error) { message.textContent = error.message; submitImport.disabled = false; }
  finally { $('#preview-import').disabled = false; }
});

showPage();
loadItems().then(() => selectItem(state.items.find(item => item.ai_result === 'NG' && !item.final_result)?.id || state.items[0]?.id)).catch(error => toast(error.message, true));
