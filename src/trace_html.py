"""A self-contained, offline viewer for a saved VASA run."""
import argparse
import base64
import html
import io
import json
import re
from pathlib import Path


def _text(message):
    content = message.get('content', [])
    if isinstance(content, str):
        return content
    return '\n'.join(p.get('text', '') for p in content if p.get('type') == 'text')


def _action(text):
    match = re.search(r'<tool>\s*(.*?)\s*</tool>', text, re.S)
    try:
        call = json.loads(match[1]) if match else {}
        name, args = call['name'], call.get('parameters', {})
        labels = {'set_strategy': 'Plan', 'segment_phrase': 'Segment',
                  'examine_each_mask': 'Inspect candidates', 'update_working_mask': 'Edit mask',
                  'return_final_output': 'Finish', 'report_no_mask': 'No target found'}
        label = labels.get(name, str(name))
        detail = ''
        if name == 'segment_phrase':
            detail = args.get('text_prompt', '')
        elif name == 'update_working_mask':
            label = str(args.get('operation', 'Edit')).capitalize()
            detail = 'Candidates ' + str(args.get('selected_masks', []))
        elif name == 'set_strategy':
            detail = args.get('text', '')
        return label, str(detail), call
    except (ValueError, KeyError, TypeError):
        verdict = re.search(r'<verdict>(.*?)</verdict>', text, re.S)
        return ('Inspect: ' + verdict[1].strip() if verdict else 'Response / recovery'), '', None


_SECTION = re.compile(r'(?:^|\n)[ \t]*(?:\d+[.)]\s*)?([A-Z][A-Za-z]{2,11})\s*:[ \t]*', re.M)
_COUNT = re.compile(r'generated\s+(\d+)\s+available\s+mask')
_GENERATION = re.compile(r'Generation\s+(\d+)')

# Which persistent panel a tool acts on. An unlisted tool still gets a step; it
# simply lights nothing, which is what keeps this general across new tools.
_PANEL = {'set_strategy': 'strategy', 'segment_phrase': 'sam3',
          'update_working_mask': 'mask', 'examine_each_mask': 'sam3'}


def _images(message):
    content = message.get('content', [])
    if not isinstance(content, list):
        return []
    return [p['image'] for p in content
            if isinstance(p, dict) and p.get('type') == 'image' and p.get('image')]


def _rationale(text):
    """The model's closing move, verbatim: last labeled section, else last sentence."""
    think = re.search(r'<think>(.*?)</think>', text, re.S)
    body = (think[1] if think else re.sub(r'<tool>.*?</tool>', '', text, flags=re.S)).strip()
    sections = list(_SECTION.finditer(body))
    if sections:
        tail = body[sections[-1].end():].strip()
        if tail:
            return ' '.join(tail.split())
    sentences = re.findall(r'[^.!?]+[.!?]', body.replace('\n', ' '))
    return ' '.join((sentences[-1] if sentences else body).split())[:200]


def _headline(text, cap=88):
    """(head, rest) split at the first sentence end, colon or semicolon."""
    joined = ' '.join((text or '').strip().split())
    stop = re.search(r'[.!?](?:\s|$)|:\s|;\s', joined)
    head = joined[:stop.start()].strip() if stop else joined
    rest = joined[stop.end():].strip() if stop else ''
    if len(head) > cap:
        cut = head.rfind(' ', 0, cap)
        rest = (head[cut:].strip() + ' ' + rest).strip()
        head = head[:cut].rstrip(' ,;:') + '…'
    elif stop and joined[stop.start()] in '.!?':
        head += '.'
    return head, rest


def _describe(call, masks_read):
    """(kind, decision) for one tool call; unknown tools fall through intact."""
    if call is None:
        return 'Unparsed reply', 'no tool call in this response'
    name, args = call.get('name'), call.get('parameters') or {}
    if name == 'set_strategy':
        return 'Strategy', _headline(args.get('text', ''))[0]
    if name == 'segment_phrase':
        return 'Tool call', 'segment_phrase("%s")' % args.get('text_prompt', '')
    if name == 'update_working_mask':
        operation = str(args.get('operation', 'edit'))
        chosen = ', '.join('#%s' % s for s in (args.get('selected_masks') or []))
        read = ('Read %d mask%s' % (masks_read, '' if masks_read == 1 else 's')
                if masks_read else 'Read results')
        return 'Reads result, decides', ('%s → %s %s' % (read, operation, chosen)).strip()
    if name == 'examine_each_mask':
        return 'Inspects candidates', 'examine_each_mask()'
    if name == 'return_final_output':
        return 'Final', 'Return final output'
    if name == 'report_no_mask':
        return 'Final', 'Report no target found'
    rendered = ', '.join('%s=%s' % (k, json.dumps(v)) for k, v in args.items())
    return str(name), ('%s(%s)' % (name, rendered))[:120]


def _tick(call):
    if call is None:
        return 'retry'
    name, args = call.get('name'), call.get('parameters') or {}
    if name == 'set_strategy':
        return 'Plan'
    if name == 'segment_phrase':
        return ((args.get('text_prompt') or '?').split() or ['?'])[-1][:12]
    if name == 'update_working_mask':
        operation = str(args.get('operation', 'edit'))
        return {'add': '+ add', 'remove': '− rm'}.get(operation, operation)[:12]
    if name in ('return_final_output', 'report_no_mask'):
        return 'Finish'
    return str(name)[:12]


def build_run(run_dir, metadata=None, history=None):
    """Turn a saved run into viewer data: {query, model, images, steps, beats}.

    Every message in the history produces at least one beat, so nothing is ever
    dropped -- warnings, retries and unrecognised tools all stay visible. Panel
    lighting and the transit beats between panels are enrichment layered onto
    the beats whose shape is recognised.
    """
    root = Path(run_dir)
    metadata = metadata if metadata is not None else json.loads((root / 'result.json').read_text())
    history = history if history is not None else json.loads((root / 'trace/history.json').read_text())

    beats, steps, images = [], [], {}
    state = {'candidate': None, 'count': None, 'mask': None, 'generation': None,
             'prompt': None, 'operation': None, 'strategy': None, 'strategy_body': None}
    pending_prompt = [None]
    seen = set()

    def key_for(reference):
        if reference in images:
            return images[reference]
        if reference.endswith('input.png'):
            key = 'INPUT'
        elif 'overlay' in reference:
            found = re.search(r'overlay_(\d+)', reference)
            key = 'MASK%s' % (found[1] if found else len(images))
        else:
            key = 'CAND%d' % sum(1 for v in images.values() if v.startswith('CAND'))
        images[reference] = key
        return key

    def emit(kind, label, light='none', step=None):
        beat = dict(kind=kind, label=label, light=light, step=step, steps=len(steps))
        beat.update(state)
        beats.append(beat)

    for index, message in enumerate(history):
        seen.add(index)
        text, pictures = _text(message), _images(message)

        if message.get('role') == 'system':
            emit('system', 'system prompt loaded')
            continue

        if message.get('role') == 'assistant':
            call = _action(text)[2]
            read = int(str(state['count']).split()[0]) if state['count'] else None
            kind, decision = _describe(call, read)
            name = (call or {}).get('name')
            if name == 'segment_phrase':
                pending_prompt[0] = 'segment_phrase · "%s"' % (
                    (call.get('parameters') or {}).get('text_prompt', ''))
            if name == 'update_working_mask':
                args = call.get('parameters') or {}
                state['operation'] = '%s %s' % (
                    str(args.get('operation', 'edit')).capitalize(),
                    ', '.join('#%s' % s for s in (args.get('selected_masks') or [])))
            panel = _PANEL.get(name) if call else None
            steps.append(dict(kind=kind, decision=decision, rationale=_rationale(text),
                              full=text.strip(), tick=_tick(call), panel=panel or '',
                              unparsed=call is None, warning=False))
            emit('decide', 'agent decides', 'agent', len(steps) - 1)
            if panel:
                emit('to' + panel.capitalize(), 'agent → ' + panel, 'agent', len(steps) - 1)
            continue

        step = len(steps) - 1 if steps else None
        if '[WARNING]' in text:
            # The controller's correction is part of the conversation the agent
            # sees, so it belongs in the transcript rather than being hidden.
            body = ' '.join(text.replace('[WARNING]', '').split())
            steps.append(dict(kind='Controller warning', decision=_headline(body, 70)[0],
                              rationale='', full=text.strip(), tick='warn', panel='',
                              unparsed=False, warning=True))
            emit('warning', 'controller warns the agent', 'none', len(steps) - 1)
        elif '[SAM3 OUTPUT]' in text:
            found = _COUNT.search(text)
            state['count'] = ('%s mask%s' % (found[1], '' if found[1] == '1' else 's')
                              if found else None)
            if pending_prompt[0]:                      # only once the result lands
                state['prompt'], pending_prompt[0] = pending_prompt[0], None
            if pictures:
                state['candidate'] = key_for(pictures[0])
            emit('sam3', 'SAM3 returns candidate masks', 'sam3', step)
            emit('return', 'SAM3 → agent', 'sam3', step)
        elif '[WORKING MASK]' in text:
            found = _GENERATION.search(text)
            if pictures:
                state['mask'] = key_for(pictures[0])
            state['generation'] = 'gen %s' % (found[1] if found else '0')
            if found and state['operation']:
                state['operation'] = '%s → generation %s' % (
                    state['operation'].split(' → ')[0], found[1])
            if found:
                emit('maskEdit', 'working mask edited', 'mask', step)
                emit('maskBack', 'working mask → agent', 'mask', step)
            else:
                emit('input', 'working mask starts empty', 'none', step)
        elif '[STRATEGY]' in text:
            state['strategy'], state['strategy_body'] = _headline(text.split(']', 1)[-1].strip())
            emit('strategySet', 'strategy saved, and it guides the rest', 'strategy', step)
        elif '[INPUT IMAGE]' in text or '[INITIAL QUERY]' in text:
            if pictures:
                key_for(pictures[0])
            emit('send', 'you send the image and the query', 'none', step)
        else:
            emit('note', ' '.join(text.split())[:64] or 'message', 'none', step)

    assert seen == set(range(len(history))), 'every message must produce a beat'
    return dict(query=metadata.get('query', ''), model=metadata.get('model', ''),
                termination=metadata.get('termination', ''),
                images=images, steps=steps, beats=beats)


def _embed(path, width=900, quality=90):
    """Trace images are ~1000px but display around 350px wide, so the originals
    cost roughly fifteen times what the page needs. Downscale to twice the
    display size and encode as WEBP: near-original resolution, so mask
    boundaries and thin structures survive, at a fraction of the bytes."""
    from PIL import Image
    try:
        picture = Image.open(path)
        picture.load()
        picture = picture.convert('RGB')
    except OSError:
        return None
    if width and picture.width > width:
        picture = picture.resize((width, round(picture.height * width / picture.width)),
                                 Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, 'WEBP', quality=quality)
    return 'data:image/webp;base64,' + base64.b64encode(buffer.getvalue()).decode()


def _assets(root, images, width=900, quality=90):
    """key -> data URI, for images that live inside the run directory."""
    assets = {}
    for reference, key in images.items():
        path = Path(reference)
        path = (root/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            continue
        if not path.is_file():
            continue
        encoded = _embed(path, width, quality)
        if encoded:
            assets[key] = encoded
    return assets


def live_payload(run_dir, metadata=None, history=None, width=900, quality=90):
    """What a running job pushes to an already-mounted viewer: the same data the
    static page carries inline, plus the images it newly needs."""
    root = Path(run_dir).resolve()
    run = build_run(root, metadata, history)
    payload = {k: run[k] for k in ('query', 'model', 'termination', 'beats', 'steps')}
    payload['run'] = root.name
    payload['assets'] = _assets(root, run['images'], width, quality)
    return payload


def feed_html(payload):
    """Payload wrapped for delivery through a Gradio HTML component.

    Base64 in a data attribute rather than a script block: gr.HTML warns about
    script tags, and replacing this element's markup cannot disturb the mounted
    viewer. Unlike a JS event handler, nothing is written back into the
    component that sent it.
    """
    if not payload:
        return ''
    blob = json.dumps(payload, ensure_ascii=False).encode()
    return '<div data-vasa-feed="%s" hidden></div>' % base64.b64encode(blob).decode()


def shell_html(width=900, quality=90):
    """The empty viewer Gradio mounts once; rounds arrive later via feed_html."""
    return render_markup(Path(__file__).resolve().parents[1], {'query': '', 'model': ''}, [],
                         live=True, width=width, quality=quality, embed=False, embedded=True)


def render_markup(run_dir, metadata=None, history=None, live=False, width=900, quality=90,
                  embed=True, embedded=False):
    """The static shell plus a JSON payload; viewer.js fills it in.

    Only images that live inside the run directory are embedded, and every
    caption is escaped here, so a trace can never inject markup into the page.
    """
    root = Path(run_dir).resolve()
    run = build_run(root, metadata, history)
    assets = _assets(root, run['images'], width, quality)

    payload = dict(run)
    payload.pop('images', None)        # references would leak absolute paths
    # Escaping "<" keeps the payload from closing the script element early.
    blob = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    bank = ''.join(f'<img data-key="{html.escape(k, quote=True)}" src="{v}" alt="">'
                   for k, v in assets.items())
    title = html.escape(run['query'] or 'VASA run')

    def panel(name, label, badge, body, caption):
        return (f'<section class="panel {name}"><div class="head"><span class="dot"></span>'
                f'<h2>{label}</h2><span class="badge" data-badge="{name}">{badge}</span></div>'
                f'{body}<p class="cap" data-cap="{name}"></p></section>')

    wires = ''.join(
        f'<div class="wire" data-wire="{i}" data-flow=""><div class="line"></div>'
        f'<span class="out">{out}</span>' + (f'<span class="back">{back}</span>' if back else '')
        + '</div>'
        for i, (out, back) in enumerate([('agent sets &rarr;', ''),
                                         ('agent calls &rarr;', '&larr; returns masks'),
                                         ('agent edits &rarr;', '&larr; new mask')]))

    return (
        f'<div class="vasa-report" data-run="{html.escape(root.name, quote=True)}" '
        f'data-live="{str(bool(live)).lower()}" data-embed="{str(bool(embedded)).lower()}">'
        + (f'<script type="application/json" data-vasa-run>{blob}</script>' if embed else '')
        + f'<div class="asset-bank" hidden>{bank}</div>'
        f'<div class="title"><b>VASA</b>: Vision Harnessing Agent for Open Ad-hoc Segmentation</div>'
        f'<div class="grid">'
        f'<section class="agent"><div class="agent-head"><span class="live-dot"></span>'
        f'<h2>{html.escape(run["model"])}</h2><span class="badge" data-phase></span></div>'
        f'<div class="chat" data-chat data-chat-scroll></div></section>'
        f'{wires}'
        + panel('strategy', 'Strategy', 'not set', '<div class="well" data-strategy></div>', '')
        + panel('sam3', 'SAM3', '&mdash;',
                '<div class="well"><img data-img="candidate" alt="SAM3 candidate masks" hidden>'
                '<span class="muted" data-empty="candidate">Waiting for a segment_phrase call&hellip;</span></div>', '')
        + panel('mask', 'Working mask', 'gen 0',
                '<div class="well"><img data-img="mask" alt="Working mask" hidden>'
                '<span class="muted" data-empty="mask">No working mask yet&hellip;</span></div>', '')
        + f'</div><footer><button type="button" data-action="play" aria-label="Pause playback">&#9208;</button>'
        f'<span class="status" data-status></span><nav class="ticks" data-ticks '
        f'aria-label="Steps in this run"></nav></footer>'
        f'<span hidden>{title}</span></div>')


def inline_css():
    return Path(__file__).with_name('viewer.css').read_text()


def viewer_js():
    return Path(__file__).with_name('viewer.js').read_text()


def render_trace(run_dir, width=900, quality=90):
    """One self-contained file: no network, no modules, no fetch, so it opens
    straight off disk."""
    body = render_markup(run_dir, width=width, quality=quality)
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>VASA · Run walkthrough</title><style>'
            'body{margin:0;background:#000;min-height:100dvh;display:flex;'
            'align-items:center;justify-content:center}'
            # Without this the flex item is as wide as its content and
            # cannot shrink, so narrow windows scroll sideways.
            'main{width:100%;min-width:0}'
            + inline_css() + '</style></head><body><main>' + body + '</main><script>'
            + viewer_js() + '\nmountVasa(document.querySelector("main"));</script></body></html>')
    target = Path(run_dir)/'trace.html'
    target.write_text(page)
    return target


_PLAIN_CSS = """
body{font-family:sans-serif;margin:1.5rem;background:#fff;color:#111;line-height:1.5}
h1,h2,h3{font-family:sans-serif}
h1{font-size:1.6rem;margin:0 0 .4rem}
h2{font-size:1.2rem;margin:1.4rem 0 .6rem}
.visual-row{display:flex;flex-wrap:wrap;gap:14px;margin:.4rem 0 .2rem}
.visual-image{margin:0;flex:0 1 auto}
.visual-image figcaption{font-size:12px;color:#666;margin-top:4px}
.visual-image img{max-width:100%;width:340px;display:block;border:1px solid #ddd;border-radius:4px}
.input-args{margin:.75rem 0 1.25rem;padding:.5rem 0}
.input-args code{word-break:break-all}
.meta{font-size:13px;color:#555;margin-bottom:16px}
.meta code{background:#eee;padding:1px 5px;border-radius:4px}
.controls{margin:0 0 16px;display:flex;gap:8px}
.controls button{padding:5px 12px;font:inherit;font-size:13px;cursor:pointer;background:#fff;
  border:1px solid #ccc;border-radius:6px}
.controls button:hover{background:#f0f0f0}
.message{margin-bottom:10px;border:1px solid #ddd;border-radius:6px;overflow:hidden;background:#fff}
.message.assistant{border-left:4px solid #b8860b}
.message.user{border-left:4px solid #4a7ebb}
.message.system{border-left:4px solid #999}
.message-header{font-size:14px;font-weight:600;padding:8px 12px;background:#f2f2f2;cursor:pointer;
  user-select:none;display:flex;align-items:center;gap:8px}
.message-header:hover{background:#e8e8e8}
.message-header::before{content:'\\25BC';font-size:9px;transition:transform .15s}
.message-header.collapsed::before{transform:rotate(-90deg)}
.message-header .tool-snippet{color:#666;font-size:12.5px;font-weight:400;font-family:ui-monospace,monospace;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.message-body{padding:12px}
.message-body.collapsed{display:none}
.block-text{margin:6px 0;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}
.block-image img{max-width:440px;width:100%;display:block;margin:8px 0;border:1px solid #ddd;border-radius:4px}
.warn{background:#fff6e5}
"""

_PLAIN_JS = """
function toggleMessage(id){
  var body=document.getElementById('body-'+id), head=document.getElementById('header-'+id);
  if(body.classList.contains('collapsed')){body.classList.remove('collapsed');head.classList.remove('collapsed');}
  else{body.classList.add('collapsed');head.classList.add('collapsed');}
}
function expandAll(){
  document.querySelectorAll('.message-body').forEach(function(b){b.classList.remove('collapsed');});
  document.querySelectorAll('.message-header').forEach(function(h){h.classList.remove('collapsed');});
}
function collapseAll(){
  document.querySelectorAll('.message-body').forEach(function(b){b.classList.add('collapsed');});
  document.querySelectorAll('.message-header').forEach(function(h){h.classList.add('collapsed');});
}
"""


def render_plain(run_dir, metadata=None, history=None, width=760, quality=88, thumb_width=560):
    """A flat, collapsible dump of every message, laid out like the SAM3 agent
    session pages: input arguments, result visualization, conversation history.

    Deliberately interprets nothing, so it cannot misreport an unfamiliar run --
    the counterpart to render_trace(), which explains a run to someone else.
    Note this one shows the system prompt and the run's paths, so it is a local
    debugging artefact rather than something to hand out.
    """
    given = Path(run_dir)
    root = given.resolve()
    metadata = metadata if metadata is not None else json.loads((root/'result.json').read_text())
    history = history if history is not None else json.loads((root/'trace/history.json').read_text())

    def embed(reference, at=None):
        path = Path(reference)
        path = (root/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return None
        if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            return None
        return _embed(path, at or width, quality)

    arguments = [('run_dir', given), ('history_path', given/'trace/history.json'),
                 ('result_path', given/'result.json'), ('out_html_path', given/'trace_plain.html')]
    if metadata.get('project_path'):
        arguments.append(('project_path', metadata['project_path']))
    argument_rows = ''.join(
        f'<div><strong>{html.escape(name)}:</strong> <code>{html.escape(str(value))}</code></div>'
        for name, value in arguments)

    pair = [('Input', embed('input.png')),
            ('Result', embed(metadata.get('overlay', 'overlay.png')) or embed('mask.png'))]
    shown = [(label, uri) for label, uri in pair if uri]
    visualization = ('<div class="visual-row">' + ''.join(
        f'<figure class="visual-image"><img alt="{label}" src="{uri}">'
        f'<figcaption>{label}</figcaption></figure>' for label, uri in shown) + '</div>'
        if shown else '<div class="meta">No result image in this run.</div>')

    blocks = []
    for index, message in enumerate(history):
        identifier = 'msg-%d' % index
        role = str(message.get('role', '?'))
        text = _text(message)
        call = _action(text)[2] if role == 'assistant' else None
        snippet = json.dumps(call, ensure_ascii=False) if call else ''
        if role == 'user' and text.startswith('['):
            snippet = text[:text.find(']') + 1] if ']' in text else ''
        warn = ' warn' if '[WARNING]' in text else ''
        pictures = ''.join(
            f'<div class="block-image"><img src="{uri}" alt="message image"></div>'
            for uri in filter(None, (embed(r, thumb_width) for r in _images(message))))
        blocks.append(
            f'<div class="message {html.escape(role)}{warn}">'
            f'<div class="message-header collapsed" id="header-{identifier}" '
            f'onclick="toggleMessage(\'{identifier}\')">'
            f'Message {index} ({html.escape(role)})'
            + (f' <span class="tool-snippet">{html.escape(snippet[:120])}</span>' if snippet else '')
            + f'</div><div class="message-body collapsed" id="body-{identifier}">'
            f'{pictures}<div class="block-text">{html.escape(text)}</div></div></div>')

    summary = ' &middot; '.join(filter(None, [
        f"<code>{html.escape(str(metadata.get('model', '?')))}</code>",
        f'{len(history)} messages',
        html.escape(str(metadata.get('termination', ''))) or '',
        f"{metadata.get('vlm_calls')} VLM calls" if metadata.get('vlm_calls') else '']))

    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>VASA Agent Session</title><style>' + _PLAIN_CSS + '</style>'
        '<script>' + _PLAIN_JS + '</script></head><body>'
        '<h1>VASA Agent Session</h1>'
        f'<div class="meta">{html.escape(metadata.get("query") or "")}</div>'
        f'<div class="meta">{summary}</div>'
        '<div class="meta input-args"><strong>Input arguments</strong>'
        f'{argument_rows}</div>'
        '<div class="controls">'
        '<button onclick="expandAll()" type="button">Expand all</button>'
        '<button onclick="collapseAll()" type="button">Collapse all</button></div>'
        '<h2>Result Visualization</h2>'
        f'{visualization}'
        '<h2>Conversation History</h2>'
        + ''.join(blocks) +
        '</body></html>')
    target = Path(run_dir)/'trace_plain.html'
    target.write_text(page)
    return target


def inline_trace(run_dir, show_explanations=False):
    """The fragment Gradio mounts. Kept markup-only: scripts injected through
    innerHTML never execute, so app.py mounts viewer.js separately."""
    return render_markup(run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an offline HTML viewer for an existing VASA run.')
    parser.add_argument('run_dir')
    parser.add_argument('--width', type=int, default=900,
                        help='max width for embedded images (0 keeps the original)')
    parser.add_argument('--quality', type=int, default=90, help='WEBP quality')
    parser.add_argument('--plain', action='store_true',
                        help='also write trace_plain.html, a flat dump of every message')
    settings = parser.parse_args()
    print(render_trace(settings.run_dir, settings.width, settings.quality))
    if settings.plain:
        print(render_plain(settings.run_dir, width=settings.width, quality=settings.quality))
