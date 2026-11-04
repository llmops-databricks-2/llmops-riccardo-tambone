"""Dependency-free chat interface; Markdown is rendered into DOM nodes, never through innerHTML."""

EXAMPLES = (
    "Quanti paper ci sono nella raccolta?",
    "Quali paper parlano di EEG?",
    "Quali sono i lavori più citati nella raccolta?",
    "Quali paper sono più simili a quello sulla causalità di Granger?",
)

HTML = (
    """<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoGraph · Assistente bibliografico</title><link rel="stylesheet" href="/ui.css"></head>
<body>
<header><div class="brand"><span class="logo" aria-hidden="true">AG</span><div><h1>AutoGraph</h1>
<p id="scope">Caricamento raccolta...</p></div></div>
<button id="reset" type="button" title="Nuova conversazione">Nuova chat</button></header>
<main id="scroller"><section id="conversation" aria-live="polite">
<div id="welcome"><h2>Esplora la tua raccolta di paper</h2>
<p>Fai domande in linguaggio naturale sui paper acquisiti da arXiv e sui lavori che citano:
trova articoli per argomento, scopri quelli simili, esplora citazioni e riferimenti.</p>
<div id="examples">"""
    + "".join(f'<button type="button" class="example">{question}</button>' for question in EXAMPLES)
    + """</div></div></section></main>
<footer><form id="chat"><label for="question" class="sr-only">Domanda sulla raccolta</label>
<textarea id="question" required maxlength="8000" rows="1" placeholder="Chiedi qualcosa sulla raccolta..."></textarea>
<button id="send" type="submit" aria-label="Invia"><svg viewBox="0 0 24 24" aria-hidden="true">
<path d="M3 20.5 21 12 3 3.5 3 10l12 2-12 2z"/></svg></button></form>
<p class="disclaimer">Le risposte possono contenere errori: verifica i paper citati. Accesso in sola lettura.</p>
</footer><script src="/ui.js" defer></script></body></html>"""
)

CSS = r""":root{--bg:#0f141b;--panel:#161d27;--line:#263142;--text:#e6edf5;--muted:#8fa1b6;--accent:#4f8cff;
--user:#2b3a52;--code:#0b1017;color-scheme:dark}
@media (prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--line:#e2e6ec;--text:#18202b;
--muted:#5f6c7b;--accent:#2f6fec;--user:#e8eefc;--code:#f1f3f6;color-scheme:light}}
*{box-sizing:border-box}html,body{height:100%}
body{margin:0;display:flex;flex-direction:column;background:var(--bg);color:var(--text);
font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.75rem 1.25rem;
border-bottom:1px solid var(--line);background:var(--panel)}
.brand{display:flex;align-items:center;gap:.75rem}.brand h1{font-size:1.05rem;margin:0}
.logo{display:grid;place-items:center;width:2.1rem;height:2.1rem;border-radius:50%;background:var(--accent);
color:#fff;font-weight:700;font-size:.8rem;flex:none}
#scope{margin:0;color:var(--muted);font-size:.78rem}
header button{background:transparent;color:var(--text);border:1px solid var(--line);border-radius:999px;
padding:.4rem .9rem;cursor:pointer}header button:hover{border-color:var(--accent)}
main{flex:1;overflow-y:auto}#conversation{max-width:52rem;margin:0 auto;padding:1.5rem 1.25rem 2rem}
#welcome{text-align:center;padding:8vh 0 0}#welcome h2{font-size:1.6rem;margin:0 0 .5rem}
#welcome p{color:var(--muted);max-width:36rem;margin:0 auto 1.5rem}
#examples{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;text-align:left}
@media (max-width:640px){#examples{grid-template-columns:1fr}}
.example{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:12px;
padding:.75rem 1rem;cursor:pointer;font:inherit;text-align:left}.example:hover{border-color:var(--accent)}
.msg{display:flex;gap:.75rem;margin:0 0 1.5rem}.msg.user{justify-content:flex-end}
.msg.user .body{background:var(--user);border-radius:18px 18px 4px 18px;padding:.6rem 1rem;max-width:80%;
white-space:pre-wrap;overflow-wrap:anywhere}
.msg.assistant .body{flex:1;min-width:0;overflow-wrap:anywhere}
.msg.assistant .logo{width:1.9rem;height:1.9rem;font-size:.7rem;margin-top:.1rem}
.body>:first-child{margin-top:0}.body>:last-child{margin-bottom:0}
.body p{margin:.4rem 0 .8rem}.body ul,.body ol{margin:.4rem 0 .8rem;padding-left:1.4rem}
.body li{margin:.25rem 0}.body h3,.body h4{margin:1rem 0 .4rem;font-size:1rem}
.body code{background:var(--code);border:1px solid var(--line);border-radius:5px;padding:0 .3rem;font-size:.88em}
.body pre{background:var(--code);border:1px solid var(--line);border-radius:8px;padding:.75rem;overflow-x:auto}
.body pre code{border:0;padding:0;background:none}
.body blockquote{margin:.5rem 0;padding-left:.8rem;border-left:3px solid var(--line);color:var(--muted)}
.body table{border-collapse:collapse;margin:.5rem 0 .9rem;display:block;overflow-x:auto}
.body th,.body td{border:1px solid var(--line);padding:.35rem .6rem;text-align:left;vertical-align:top}
.body th{background:var(--panel)}.body a{color:var(--accent)}
.body hr{border:0;border-top:1px solid var(--line);margin:1rem 0}
.msg.error .body{color:#ff8a8a}
details{margin-top:.6rem;font-size:.8rem;color:var(--muted)}summary{cursor:pointer}
details pre{white-space:pre-wrap;max-height:22rem;overflow:auto;background:var(--code);padding:.6rem;
border-radius:8px;border:1px solid var(--line)}
.body.typing span{display:inline-block;width:.45rem;height:.45rem;margin-right:.25rem;border-radius:50%;
background:var(--muted);animation:blink 1.2s infinite both}.body.typing span:nth-child(2){animation-delay:.2s}
.body.typing span:nth-child(3){animation-delay:.4s}@keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}
footer{padding:.5rem 1.25rem 1rem}
#chat{max-width:52rem;margin:0 auto;display:flex;align-items:flex-end;gap:.5rem;background:var(--panel);
border:1px solid var(--line);border-radius:24px;padding:.45rem .45rem .45rem 1rem}
#chat:focus-within{border-color:var(--accent)}
#question{flex:1;resize:none;border:0;outline:0;background:transparent;color:inherit;font:inherit;
max-height:12rem;padding:.4rem 0}
#send{flex:none;width:2.4rem;height:2.4rem;border-radius:50%;border:0;background:var(--accent);cursor:pointer;
display:grid;place-items:center}#send svg{width:1.1rem;height:1.1rem;fill:#fff}
#send:disabled{opacity:.4;cursor:default}
.disclaimer{text-align:center;color:var(--muted);font-size:.72rem;margin:.5rem 0 0}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}"""

JS = r"""'use strict';
const history = [];
const conversation = document.querySelector('#conversation');
const scroller = document.querySelector('#scroller');
const input = document.querySelector('#question');
const send = document.querySelector('#send');
const reset = document.querySelector('#reset');
const welcome = document.querySelector('#welcome');
let busy = false;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Inline Markdown: code, bold, italic, links. Only text nodes and whitelisted elements are created.
const INLINE = /(`[^`]+`)|(\*\*[^*]+\*\*|__[^_]+__)|(\*[^*\s][^*]*\*|(?<!\w)_[^_\s][^_]*_(?!\w))|(\[[^\]]+\]\([^)\s]+\))|(https?:\/\/[^\s)]+)/;
function inline(text, parent) {
  while (text) {
    const match = INLINE.exec(text);
    if (!match) { parent.append(text); return; }
    if (match.index) parent.append(text.slice(0, match.index));
    const token = match[0];
    if (match[1]) parent.append(el('code', '', token.slice(1, -1)));
    else if (match[2]) inline(token.slice(2, -2), parent.appendChild(el('strong')));
    else if (match[3]) inline(token.slice(1, -1), parent.appendChild(el('em')));
    else {
      const parts = match[4] ? /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(token) : [token, token, token];
      if (/^https?:\/\//i.test(parts[2])) {
        const link = el('a'); link.href = parts[2]; link.target = '_blank'; link.rel = 'noopener noreferrer';
        inline(parts[1], link); parent.append(link);
      } else parent.append(parts[1]);
    }
    text = text.slice(match.index + token.length);
  }
}

const LIST = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const TABLE_RULE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
function cells(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
}
function markdown(source, parent) {
  const lines = source.replace(/\r\n?/g, '\n').split('\n');
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (/^\s*```/.test(line)) {
      const code = [];
      for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) code.push(lines[i]);
      i++;
      parent.appendChild(el('pre')).append(el('code', '', code.join('\n')));
      continue;
    }
    const heading = /^\s*(#{1,6})\s+(.*)$/.exec(line);
    if (heading) { inline(heading[2], parent.appendChild(el(heading[1].length <= 2 ? 'h3' : 'h4'))); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { parent.append(el('hr')); i++; continue; }
    if (line.includes('|') && i + 1 < lines.length && TABLE_RULE.test(lines[i + 1])) {
      const table = parent.appendChild(el('table'));
      const head = table.appendChild(el('thead')).appendChild(el('tr'));
      cells(line).forEach(cell => inline(cell, head.appendChild(el('th'))));
      const body = table.appendChild(el('tbody'));
      for (i += 2; i < lines.length && lines[i].includes('|') && lines[i].trim(); i++) {
        const row = body.appendChild(el('tr'));
        cells(lines[i]).forEach(cell => inline(cell, row.appendChild(el('td'))));
      }
      continue;
    }
    if (/^\s*>/.test(line)) {
      const quote = [];
      for (; i < lines.length && /^\s*>/.test(lines[i]); i++) quote.push(lines[i].replace(/^\s*>\s?/, ''));
      markdown(quote.join('\n'), parent.appendChild(el('blockquote')));
      continue;
    }
    if (LIST.test(line)) {
      const stack = [];
      for (; i < lines.length; i++) {
        const item = LIST.exec(lines[i]);
        if (!item) {
          if (lines[i].trim() && stack.length && /^\s+/.test(lines[i])) {
            const last = stack[stack.length - 1].list.lastChild;
            last.append(document.createElement('br')); inline(lines[i].trim(), last); continue;
          }
          break;
        }
        const indent = item[1].replace(/\t/g, '  ').length;
        const ordered = /\d/.test(item[2]);
        while (stack.length && indent < stack[stack.length - 1].indent) stack.pop();
        let top = stack[stack.length - 1];
        if (!top || indent > top.indent) {
          const list = el(ordered ? 'ol' : 'ul');
          if (ordered && parseInt(item[2], 10) > 1) list.start = parseInt(item[2], 10);
          (top ? top.list.lastChild : parent).append(list);
          top = {indent, list}; stack.push(top);
        }
        inline(item[3], top.list.appendChild(el('li')));
      }
      continue;
    }
    const paragraph = parent.appendChild(el('p'));
    let first = true;
    for (; i < lines.length && lines[i].trim() && !LIST.test(lines[i]) && !/^\s*(#{1,6}\s|```|>)/.test(lines[i]); i++) {
      if (!first) paragraph.append(document.createElement('br'));
      inline(lines[i].trim(), paragraph); first = false;
    }
  }
}

function scrollDown() { scroller.scrollTop = scroller.scrollHeight; }
function message(role, text) {
  welcome.hidden = true;
  const row = el('div', 'msg ' + role);
  const body = el('div', 'body');
  if (role !== 'user') row.append(el('span', 'logo', 'AG'));
  row.append(body); conversation.append(row);
  if (role === 'assistant') markdown(text, body); else body.textContent = text;
  scrollDown();
  return body;
}
function technical(body, data) {
  const details = el('details');
  details.append(el('summary', '', 'Dettagli tecnici'));
  details.append(el('div', '', data.tool_calls + ' chiamate tool · ' + data.tokens + ' token · trace ' + data.trace_id));
  if (data.evidence && data.evidence.length) details.append(el('pre', '', JSON.stringify(data.evidence, null, 2)));
  body.append(details);
}
function resize() { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 192) + 'px'; }

fetch('/api/collection').then(async response => {
  if (!response.ok) throw new Error('Impossibile leggere la raccolta');
  const data = await response.json();
  const scope = document.querySelector('#scope');
  scope.textContent = 'Assistente bibliografico · grafo con ' + data.node_count + ' lavori e ' + data.edge_count + ' citazioni';
  scope.title = 'Run ' + data.run_id + ' · build ' + data.graph_build_id;
}).catch(error => { document.querySelector('#scope').textContent = error.message; });

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = send.disabled = reset.disabled = true;
  input.value = ''; resize();
  const messages = [...history, {role: 'user', content: question}];
  message('user', question);
  const pending = message('typing', '');
  pending.className = 'body typing'; pending.append(el('span'), el('span'), el('span'));
  try {
    const response = await fetch('/api/chat', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Richiesta fallita');
    history.push({role: 'user', content: question}, {role: 'assistant', content: data.answer});
    while (history.length > 10 || history.reduce((n, m) => n + m.content.length, 0) > 8000) history.splice(0, 2);
    pending.parentElement.remove();
    technical(message('assistant', data.answer), data);
  } catch (error) {
    pending.parentElement.remove();
    message('error', error.message);
  } finally {
    busy = send.disabled = reset.disabled = false; input.focus(); scrollDown();
  }
}

document.querySelector('#chat').onsubmit = event => { event.preventDefault(); ask(input.value.trim()); };
input.addEventListener('input', resize);
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); ask(input.value.trim()); }
});
document.querySelectorAll('.example').forEach(button => { button.onclick = () => ask(button.textContent); });
reset.onclick = () => {
  history.length = 0;
  conversation.querySelectorAll('.msg').forEach(node => node.remove());
  welcome.hidden = false; input.focus();
};
"""
