"""A self-contained, offline viewer for a saved VASA run."""
import argparse
import base64
import html
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


def render_markup(run_dir, metadata=None, history=None, live=False):
    root = Path(run_dir).resolve()
    metadata = metadata if metadata is not None else json.loads((root/'result.json').read_text())
    history = history if history is not None else json.loads((root/'trace/history.json').read_text())
    images = {}

    def picture(reference, caption):
        path = Path(reference)
        path = (root/path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            return ''
        if not path.is_file():
            return '<p class="muted">Image unavailable</p>'
        if path not in images:
            mime = 'jpeg' if path.suffix.lower() in {'.jpg', '.jpeg'} else path.suffix[1:]
            images[path] = f'data:image/{mime};base64,' + base64.b64encode(path.read_bytes()).decode()
        asset = list(images).index(path)
        kind = 'candidates' if caption == 'SAM3 candidates' else 'inspection' if caption == 'Inspection view' else 'original' if caption == 'Original image' else 'working'
        return f'<figure data-kind="{kind}"><span data-image="{asset}" data-caption="{html.escape(caption, quote=True)}"></span><figcaption>{html.escape(caption)}</figcaption></figure>'

    steps = []
    pending = []
    for message in history:
        if message.get('role') == 'system':
            continue
        if message.get('role') == 'assistant':
            text = _text(message)
            label, detail, call = _action(text)
            step = {'label': label, 'detail': detail, 'call': call, 'text': text, 'images': pending, 'warnings': []}
            steps.append(step)
            pending = []
        else:
            text = _text(message)
            content = message.get('content', [])
            refs = [p['image'] for p in content if isinstance(p, dict) and p.get('type') == 'image'] if isinstance(content, list) else []
            caption = 'Working mask' if '[WORKING MASK]' in text else 'SAM3 candidates' if '[SAM3 OUTPUT]' in text else 'Inspection view'
            # Auxiliary inspection inputs precede their assistant verdict.
            if refs and '[INPUT IMAGE]' not in text and '[WORKING MASK]' not in text and '[SAM3 OUTPUT]' not in text:
                pending.extend((ref, caption) for ref in refs)
            elif steps:
                steps[-1]['images'].extend((ref, caption) for ref in refs)
            if steps and '[WARNING]' in text:
                steps[-1]['warnings'].append(text)

    cards = []
    navigation = []
    working = 'input.png'
    working_caption = 'Original image'
    for index, step in enumerate(steps, 1):
        detail = html.escape(step['detail'])
        if step['label'] == 'Plan' and detail:
            detail = f'<details><summary>View plan</summary><p>{detail}</p></details>'
        else:
            detail = f'<p class="detail">{detail}</p>' if detail else ''
        for ref, caption in step['images']:
            if caption == 'Working mask':
                working = ref
                working_caption = "Working mask"
        navigation.append(f'<button data-select="{index-1}" aria-pressed="false">{index:02} · {html.escape(step["label"])} {html.escape(step["detail"][:60])}</button>')
        visuals = ''.join(picture(ref, caption) for ref, caption in step['images'])
        warnings = ''.join(f'<p class="warning">{html.escape(w)}</p>' for w in step['warnings'])
        raw = html.escape(json.dumps(step['call'], indent=2)) if step['call'] else ''
        args = f'<details><summary>Tool arguments</summary><pre>{raw}</pre></details>' if raw else ''
        explanation = re.sub(r'<tool>.*?</tool>', '', step['text'], flags=re.S).replace('<think>', '').replace('</think>', '').strip()
        explain = f'<details class="explanation"><summary>Model explanation</summary><pre>{html.escape(explanation)}</pre></details>' if explanation else ''
        cards.append(f'<article data-step="{index-1}" id="step-{index}" hidden><div class="working-state" hidden>{picture(working, working_caption)}</div><header><span class="number">{index:02}</span><h3>{html.escape(step["label"])}</h3></header>{detail}<div class="gallery">{visuals}</div>{warnings}{args}{explain}</article>')
    original = picture('input.png', 'Original image')
    final = picture(metadata.get('overlay', 'overlay.png'), 'Final result') if not live else picture(working, 'Current working mask')
    query = html.escape(metadata.get('query', 'Saved run'))
    meta = html.escape(f"{metadata.get('model', '')} · {'Running' if live else metadata.get('termination', 'unknown').replace('_', ' ')}")
    run_id = html.escape(root.name, quote=True)
    bank = '<div class="asset-bank" hidden>' + ''.join(f'<img data-asset="{i}" src="{uri}" alt="Saved view">' for i, uri in enumerate(images.values())) + '</div>'
    return f'''<div class="vasa-report" data-run="{run_id}" data-live="{str(live).lower()}" data-revision="{len(history)}-{live}">
{bank}
<div class="brand">VASA / VISUAL CONSTRUCTION</div><h1>{query}</h1><div class="meta">{meta}</div>
<div class="workspace"><section class="canvas" aria-label="Segmentation view">
<div class="view-status"><strong class="view-badge">Final result</strong><span class="view-description"></span></div><div class="stage">{final or original}</div><div class="controls"><button data-action="mask">Show original</button><button data-action="working">Working mask</button><button data-action="step-view">Step output</button><button data-action="previous">Previous</button><button data-action="next">Next</button><span class="selection-label">Final result</span></div>
<div class="playback" {'hidden' if live else ''}><button data-action="play">Play</button><input type="range" min="0" max="{len(steps)}" value="{len(steps)}" step="1" aria-label="Replay step"><span class="playback-hint">3 seconds per step</span></div>
<div class="original" hidden>{original}</div><div class="final-image" hidden>{final}</div></section>
<aside class="activity" aria-label="Model activity"><h2>How the mask was built</h2>
<div class="controls"><button data-select="final">{'Current mask' if live else 'Final result'}</button><button data-action="follow" {'hidden' if not live else ''}>Following live</button></div>
<nav class="step-list" aria-label="Choose a step">{''.join(navigation)}</nav>
<div class="final-detail"><p>{'Working on your query. Completed actions appear here.' if live else 'Select an action to inspect its images and decision.'}</p></div>
{''.join(cards)}</aside></div>
<footer>{len(steps)} responses · Select a step to explore · Explanations are optional</footer></div>'''


def inline_css():
    return Path(__file__).with_name('viewer.css').read_text()


def viewer_js():
    return Path(__file__).with_name('viewer.js').read_text()


def render_trace(run_dir):
    body = render_markup(run_dir)
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VASA · Run walkthrough</title><style>body{margin:0;padding:20px;background:#f4f6f2}' + inline_css() + '</style></head><body><main>' + body + '</main><script>' + viewer_js() + '\nmountVasa(document.querySelector("main"));</script></body></html>'
    target = Path(run_dir)/'trace.html'
    target.write_text(page)
    return target


def inline_trace(run_dir, show_explanations=False):
    body = render_markup(run_dir)
    if show_explanations:
        body = body.replace('<details class="explanation">', '<details class="explanation" open>')
    return body


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build an offline HTML viewer for an existing VASA run.')
    parser.add_argument('run_dir')
    print(render_trace(parser.parse_args().run_dir))
