"""Small dependency-free chat interface; evidence is rendered as text, never executable paper HTML."""

HTML = """<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoGraphRS · Assistente bibliografico</title><link rel="stylesheet" href="/ui.css"></head>
<body><main><h1>AutoGraphRS</h1><p>Assistente bibliografico read-only · Gateway, pgvector e Genie</p>
<p id="scope">Caricamento raccolta...</p><section id="conversation" aria-live="polite"></section>
<form id="chat"><label for="question">Domanda sulla raccolta</label>
<textarea id="question" required maxlength="8000" rows="3"></textarea>
<button id="send" type="submit">Invia</button><button id="reset" type="button">Nuova conversazione</button>
</form><p id="status" role="status"></p>
<p>Le risposte possono contenere errori: controlla le evidenze. Nessuna acquisizione o modifica dei dati.</p>
</main><script src="/ui.js" defer></script></body></html>"""

CSS = """body{font:16px system-ui;background:#101824;color:#e5edf7;margin:0}
main{max-width:900px;margin:auto;padding:2rem}textarea{box-sizing:border-box;width:100%;margin:.5rem 0;
background:#182536;color:inherit;padding:1rem;border:1px solid #6b8199;border-radius:8px}
button{padding:.7rem 1rem;margin-right:.5rem;cursor:pointer}article{padding:1rem;background:#182536;
margin:1rem 0;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere}pre{white-space:pre-wrap}
#scope{color:#a7c7ed;font-size:.85rem}label{display:block}details{margin-top:1rem}"""

JS = """'use strict';
const history = [];
const status = document.querySelector('#status');
const conversation = document.querySelector('#conversation');
function add(role, text, evidence) {
  const article = document.createElement('article');
  const title = document.createElement('strong');
  title.textContent = role + '\\n'; article.append(title, document.createTextNode(text));
  if (evidence && evidence.length) {
    const details = document.createElement('details'); const summary = document.createElement('summary');
    summary.textContent = 'Evidenze restituite dai tool'; const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(evidence, null, 2); details.append(summary, pre); article.append(details);
  }
  conversation.append(article);
}
fetch('/api/collection').then(async response => {
  if (!response.ok) throw new Error('Impossibile leggere la raccolta');
  const data = await response.json();
  document.querySelector('#scope').textContent = 'Run: ' + data.run_id + ' · Build: ' + data.graph_build_id;
}).catch(error => {status.textContent = error.message;});
document.querySelector('#reset').onclick = () => {history.length = 0; conversation.replaceChildren(); status.textContent = '';};
document.querySelector('#chat').onsubmit = async event => {
  event.preventDefault();
  const question = document.querySelector('#question').value.trim(); if (!question) return;
  const send = document.querySelector('#send'); const reset = document.querySelector('#reset');
  send.disabled = reset.disabled = true; status.textContent = 'Ricerca e verifica delle evidenze...';
  const messages = [...history, {role: 'user', content: question}];
  add('Tu', question);
  try {
    const response = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages})});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Richiesta fallita');
    history.push({role:'user', content:question}, {role:'assistant', content:data.answer});
    while (history.length > 10 || history.reduce((n,m) => n + m.content.length,0) > 8000) history.splice(0,2);
    add('Assistente', data.answer, data.evidence); document.querySelector('#question').value = '';
    status.textContent = data.tool_calls + ' chiamate tool · ' + data.tokens + ' token · trace ' + data.trace_id;
  } catch (error) {status.textContent = error.message;}
  finally {send.disabled = reset.disabled = false;}
};
"""
