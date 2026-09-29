"""Local browser interface for VASA. Run with python app.py."""
import os
import json
from queue import Queue
from threading import Thread
from datetime import datetime
from pathlib import Path
from src.runs import automatic_output

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
import gradio as gr
from dotenv import load_dotenv
from src.trace_html import shell_html, feed_html, live_payload, inline_css, viewer_js, render_trace, render_plain

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT/'outputs'
load_dotenv(ROOT/'.env', override=False)
QUERIES = ["segment the cat's head without what she uses to hear and see",
           'What parts are about to make contact', 'Segment everything with a striped pattern']


def saved_runs():
    entries = []
    for record in OUTPUTS.glob('*/result.json'):
        if not (record.parent/'trace/history.json').is_file():
            continue
        try:
            data = json.loads(record.read_text())
            created = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.fromtimestamp(record.stat().st_mtime).astimezone()
            query = ' '.join(data.get('query', 'Saved run').split())
            short = query[:48] + ('…' if len(query) > 48 else '')
            label = f"{data.get('image_name', record.parent.name)} · {short} · {created:%b %d, %Y %H:%M:%S}"
            entries.append((created.timestamp(), label, record.parent.name))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return [(label, name) for _, label, name in sorted(entries, reverse=True)]


def run_folder(name):
    folder = (OUTPUTS/str(name)).resolve()
    if not name or folder.parent != OUTPUTS.resolve() or not (folder/'result.json').is_file():
        raise gr.Error('Select a saved VASA run.')
    return folder


def show_run(name, explanations):
    if not name:
        return '', None, None, None
    folder = run_folder(name)
    render_trace(folder)
    render_plain(folder)
    return (feed_html(live_payload(folder)), str(folder/'mask.png'),
            str(folder/'overlay.png'), str(folder/'trace.html'))


def run(image, query, device, explanations, progress=gr.Progress(), preview_callback=None):
    if not image or not query.strip():
        raise gr.Error('Choose an image and enter a query.')
    if not all(os.getenv(key) for key in ('VASA_BASE_URL', 'VASA_MODEL', 'VASA_API_KEY')):
        raise gr.Error('Set VASA_BASE_URL, VASA_MODEL, and VASA_API_KEY in .env first.')
    from src.inference import segment
    folder = automatic_output(image, query, OUTPUTS)
    name = folder.name
    progress(0, desc='Loading SAM3 and preparing the image…')
    def status(text):
        if 'Round ' in text:
            progress(None, desc=text.strip().strip('-').strip())
    sent = set()
    def push(snapshot):
        payload = live_payload(folder, {'query': query, 'model': os.getenv('VASA_MODEL', '')}, snapshot)
        # The viewer keeps the images it has already been given, so only ship
        # the ones it has not seen; otherwise every round resends the lot.
        payload['assets'] = {k: v for k, v in payload['assets'].items() if k not in sent}
        sent.update(payload['assets'])
        preview_callback(feed_html(payload))
    try:
        result = segment(image, query, folder, device=device, progress_callback=status,
                         preview_callback=push if preview_callback else None)
    except Exception as exc:
        # Do not expose provider exception bodies or credentials in the browser.
        raise gr.Error(f'Run failed ({type(exc).__name__}). See the local terminal for details.') from exc
    view, mask, overlay, report = show_run(name, explanations)
    return view, mask, overlay, report, name, gr.update(choices=saved_runs(), value=name, interactive=True), f"{result['termination'].replace('_', ' ').capitalize()} · {result['vlm_calls']} VLM calls"


def stream_run(image, query, device, explanations):
    events = Queue()
    def work():
        try:
            result = run(image, query, device, explanations,
                         progress=lambda *a, **kw: events.put(('status', kw.get('desc', 'Running…'))),
                         preview_callback=lambda payload: events.put(('preview', payload)))
            events.put(('done', result))
        except Exception as exc:
            events.put(('error', exc))
    worker = Thread(target=work, daemon=True)
    worker.start()
    try:
        yield '', None, None, None, '', gr.update(interactive=False), 'Preparing your image…'
        while True:
            kind, value = events.get()
            if kind == 'done':
                yield value
                return
            if kind == 'error':
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(interactive=True), 'Run failed. See the terminal for details.'
                raise value
            if kind == 'preview':
                yield value, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), 'Running · following completed actions'
            else:
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), value
    finally:
        worker.join()


def build_app():
    import torch
    with gr.Blocks(title='VASA · Visual construction') as app:
        gr.Markdown('# VASA\nDescribe a region. Follow how the mask is constructed.')
        with gr.Accordion('Image and query', open=True) as composer:
            with gr.Row():
                image = gr.Image(value=str(ROOT/'examples/pipi.png'), type='filepath', label='Your image', height=340)
                with gr.Column():
                    query = gr.Textbox(value=QUERIES[0], label='What would you like to segment?', lines=3)
                    gr.Examples(examples=[[q] for q in QUERIES], inputs=query, label='Try a query')
                    with gr.Accordion('Settings', open=False):
                        device = gr.Radio(['cpu','cuda'], value='cuda' if torch.cuda.is_available() else 'cpu', label='SAM3 device')
                        gr.Markdown('VLM endpoint, model, and key are loaded from your `.env` file.')
                    start = gr.Button('Run VASA', variant='primary')
                    status = gr.Markdown('Ready')
        with gr.Row():
            runs = gr.Dropdown(choices=saved_runs(), value=None, label='Review a saved run')
            refresh = gr.Button('Refresh runs')
            explanations = gr.State(False)
        selected = gr.State('')
        # Mounted once with an empty shell. Rounds arrive as data through
        # `feed` below, so the DOM is never replaced and playback, scroll
        # position and in-flight animation all survive each update.
        viewer = gr.HTML(shell_html(), elem_id='vasa-playground', min_height=240,
                         apply_default_css=False,
                         js_on_load=viewer_js() + '\nmountVasa(element);', autoscroll=False)
        # Rounds land here as an inert data attribute. Swapping this element's
        # markup leaves the mounted viewer alone, and nothing is written back
        # into the component that sent it.
        feed = gr.HTML('', elem_id='vasa-feed')
        with gr.Row():
            mask = gr.File(label='Download mask', interactive=False)
            overlay = gr.File(label='Download overlay', interactive=False)
            report = gr.File(label='Download HTML report', interactive=False)
        # Collapsing the composer moves the viewer up the page, so wait a frame
        # for that to settle before scrolling to it.
        reveal = """() => { requestAnimationFrame(() => requestAnimationFrame(() => {
            const playground = document.getElementById('vasa-playground');
            if (playground) {
                playground.setAttribute('tabindex', '-1');
                playground.focus({preventScroll: true});
                playground.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
        })); }"""
        start.click(lambda: gr.update(open=False), outputs=composer, queue=False).then(fn=None, js=reveal)
        runs.input(lambda: gr.update(open=False), outputs=composer, queue=False).then(fn=None, js=reveal)
        start.click(stream_run, [image,query,device,explanations], [feed,mask,overlay,report,selected,runs,status], concurrency_limit=1, concurrency_id='vasa-model', trigger_mode='once')
        def review(name, expanded):
            return (*show_run(name, expanded), name)
        runs.input(review, [runs, explanations], [feed,mask,overlay,report,selected])
        refresh.click(lambda:gr.update(choices=saved_runs()), outputs=runs)
    return app


if __name__ == '__main__':
    build_app().queue().launch(server_name='127.0.0.1', share=False, css=inline_css(),
                             blocked_paths=[str(ROOT/'.env'), str(ROOT/'.git')])
