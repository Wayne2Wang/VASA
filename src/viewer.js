// One viewer for offline reports and the live Gradio run.
// Python renders a static shell plus a JSON payload from build_run(); this
// fills it in and walks the beats. Every string reaches the DOM through
// textContent, so trace text can never become markup.
function mountVasa(host) {
  const HOLD = {system: 0.7, send: 1.5, input: 0.9, decide: 1.6, toStrategy: 1,
                strategySet: 1.45, toSam3: 1, sam3: 1.25, return: 1.6, toMask: 1,
                maskEdit: 1.25, maskBack: 1.6, warning: 1.5, note: 1};
  const BASE = 1300;
  let root = null, data = null, beat = 0, playing = false, open = null, stopAt = null, timer = null;

  const el = (selector) => root && root.querySelector(selector);
  const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

  function read() {
    const holder = host.querySelector('.vasa-report');
    if (!holder) return null;
    const blob = holder.querySelector('script[type="application/json"][data-vasa-run]');
    try { return blob ? JSON.parse(blob.textContent) : null; } catch (error) { return null; }
  }

  function source(key) {
    const asset = key && root.querySelector('.asset-bank img[data-key="' + key + '"]');
    return asset ? asset.getAttribute('src') : '';
  }

  function stop() { if (timer !== null) clearTimeout(timer); timer = null; }

  function schedule() {
    stop();
    if (!playing) return;
    const limit = stopAt === null ? data.beats.length - 1 : stopAt;
    if (beat >= limit) { playing = false; draw(); return; }
    const hold = HOLD[data.beats[beat].kind] || 1;
    timer = setTimeout(() => {
      if (!host.isConnected) { stop(); return; }
      beat += 1; draw(); schedule();
    }, Math.round(BASE * hold));
  }

  function goTo(step) {
    const first = data.beats.findIndex((b) => b.step === step);
    if (first < 0) return;
    let last = data.beats.length - 1;
    for (let i = first; i < data.beats.length; i++) {
      if (data.beats[i].step !== null && data.beats[i].step > step) { last = i - 1; break; }
    }
    beat = first; stopAt = last; playing = true; open = null;
    draw(); schedule();
  }

  function toggleRun() {
    playing = !playing;
    if (playing) {
      stopAt = null; open = null;
      if (beat >= data.beats.length - 1) beat = 0;
    }
    draw(); schedule();
  }

  function toggleBox(index) {
    stop();
    const opening = open !== index;
    open = opening ? index : null;
    playing = false;
    draw();
    if (!opening) return;
    const chat = el('[data-chat]');
    const box = chat && chat.querySelector('[data-box="' + index + '"]');
    // Expanding the last box would otherwise open it below the fold.
    if (chat && box) chat.scrollTop += box.getBoundingClientRect().top - chat.getBoundingClientRect().top - 12;
  }

  function stepBox(step, index) {
    const wrap = document.createElement('div');
    wrap.className = 'box' + (step.warning ? ' warn' : '') + (step.unparsed ? ' unparsed' : '');
    wrap.dataset.box = String(index);

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'box-head';
    button.dataset.box = String(index);
    button.setAttribute('aria-expanded', String(open === index));

    const top = document.createElement('span');
    top.className = 'box-top';
    const kind = document.createElement('span');
    kind.className = 'box-kind';
    kind.textContent = step.kind;
    const pill = document.createElement('span');
    pill.className = 'box-pill';
    pill.textContent = (open === index ? '▴ hide full message' : '▾ full message');
    top.append(kind, pill);

    const decision = document.createElement('span');
    decision.className = 'box-decision';
    decision.textContent = step.decision;
    button.append(top, decision);

    if (step.rationale) {
      const why = document.createElement('span');
      why.className = 'box-why';
      why.textContent = step.rationale;
      button.append(why);
    }
    wrap.append(button);

    if (open === index) {
      const full = document.createElement('pre');
      full.className = 'box-full';
      full.textContent = step.full;
      wrap.append(full);
    }
    return wrap;
  }

  function draw() {
    if (!root || !data) return;
    const beats = data.beats, steps = data.steps;
    const current = beats[Math.min(beat, beats.length - 1)] || {};
    // Controls belong to a run you can scrub. A run still in progress has no
    // end to seek to; one that has terminated does, saved or just finished.
    const finished = !!data.termination;
    const kind = current.kind;

    const acting = current.light === 'agent';
    const lit = {
      strategy: kind === 'strategySet' || (current.strategy && acting),
      sam3: kind === 'sam3' || kind === 'return',
      mask: kind === 'maskEdit' || kind === 'maskBack',
    };
    root.dataset.agent = acting ? 'on' : 'off';
    root.dataset.strategy = lit.strategy ? (kind === 'strategySet' ? 'set' : 'guiding') : 'off';
    root.dataset.sam3 = lit.sam3 ? 'on' : 'off';
    root.dataset.mask = lit.mask ? 'on' : 'off';
    const flow = {toStrategy: '0r', toSam3: '1r', return: '1l', toMask: '2r', maskBack: '2l'}[kind] || '';
    root.querySelectorAll('[data-wire]').forEach((wire) => {
      const want = wire.dataset.wire;
      wire.dataset.flow = flow.startsWith(want) ? flow.slice(1) : '';
    });

    // panels
    const strategy = el('[data-strategy]');
    strategy.replaceChildren();
    if (current.strategy) {
      const head = document.createElement('div');
      head.className = 'strategy-head';
      head.textContent = current.strategy;
      strategy.append(head);
      if (current.strategy_body) {
        const body = document.createElement('div');
        body.className = 'strategy-body';
        body.textContent = current.strategy_body;
        strategy.append(body);
      }
    } else {
      const none = document.createElement('span');
      none.className = 'muted';
      none.textContent = 'Not set yet.';
      strategy.append(none);
    }
    el('[data-badge="strategy"]').textContent = current.strategy ? 'persistent' : 'not set';

    // Only ever set a src we actually have; an <img> with an empty src renders
    // as a broken image with its alt text showing.
    function show(selector, key) {
      const picture = el('[data-img="' + selector + '"]');
      const uri = source(key);
      if (uri) picture.src = uri;
      picture.hidden = !uri;
      const empty = el('[data-empty="' + selector + '"]');
      if (empty) empty.hidden = !!uri;
      return uri;
    }

    show('candidate', current.candidate);
    el('[data-badge="sam3"]').textContent = current.count || '—';
    el('[data-cap="sam3"]').textContent = current.prompt || 'No call yet';

    show('mask', current.mask || 'MASK0');
    el('[data-badge="mask"]').textContent = current.generation || 'gen 0';
    el('[data-cap="mask"]').textContent = current.operation || 'Working mask · empty';
    el('[data-phase]').textContent = acting ? 'deciding'
      : (kind === 'send' || kind === 'system' || kind === 'input') ? 'reading your message'
      : 'waiting on tools';

    // conversation, opening with what was actually asked
    const chat = el('[data-chat]');
    const asked = document.createElement('div');
    asked.className = 'asked';
    const bubble = document.createElement('div');
    bubble.className = 'asked-bubble';
    const thumb = source('INPUT');
    if (thumb) {
      const picture = document.createElement('img');
      picture.src = thumb;
      picture.alt = 'Input image';
      bubble.append(picture);
    }
    const query = document.createElement('p');
    query.textContent = '“' + (data.query || '(no query recorded)') + '”';
    bubble.append(query);
    asked.append(bubble);
    const shown = steps.slice(0, current.steps || 0);
    chat.replaceChildren(asked, ...shown.map((step, i) => {
      const node = stepBox(step, i);
      if (open !== i && i !== shown.length - 1) node.classList.add('faded');
      return node;
    }));
    const next = beats[beat + 1];
    if (next && next.steps > (current.steps || 0)) {
      const dots = document.createElement('div');
      dots.className = 'typing';
      dots.append(...[0, 1, 2].map(() => document.createElement('i')));
      chat.append(dots);
    }
    if (open === null || playing) chat.scrollTop = chat.scrollHeight;

    // footer
    const play = el('[data-action="play"]');
    if (play) {
      play.textContent = playing ? '⏸' : beat >= beats.length - 1 ? '↻' : '▶';
      play.setAttribute('aria-label', playing ? 'Pause playback'
        : beat >= beats.length - 1 ? 'Replay from the start' : 'Resume playback');
      play.hidden = !finished;
    }
    const position = current.step === null || current.step === undefined ? 1 : current.step + 1;
    el('[data-status]').textContent =
      steps.length ? position + '/' + steps.length + ' · ' + (current.label || '') : (current.label || '');

    const ticks = el('[data-ticks]');
    ticks.replaceChildren(...steps.map((step, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'tick';
      button.dataset.step = String(i);
      button.textContent = step.tick;
      button.title = 'Step ' + (i + 1) + ': ' + step.kind;
      button.setAttribute('aria-pressed', String(i === current.step));
      return button;
    }));
    ticks.hidden = !finished;
  }

  function load(next) {
    data = next;
    if (!data || !data.beats || !data.beats.length) return;
    beat = root.dataset.live === 'true' ? data.beats.length - 1 : 0;
    playing = root.dataset.live !== 'true' && !reduced();
    stopAt = null; open = null;
    draw();
    schedule();
  }

  host.addEventListener('click', (event) => {
    const target = event.composedPath().find((n) => n instanceof Element && n.closest('.vasa-report'));
    if (!target || !data) return;
    const button = target.closest('button');
    if (!button) return;
    if (button.dataset.action === 'play') { toggleRun(); return; }
    if (button.dataset.step !== undefined) { goTo(Number(button.dataset.step)); return; }
    if (button.dataset.box !== undefined) { toggleBox(Number(button.dataset.box)); }
  }, true);

  // Live runs push new data instead of replacing the DOM, so playback state,
  // scroll position and in-flight animation survive each round.
  window.vasaPush = function (incoming) {
    if (!root) return;
    const payload = typeof incoming === 'string' ? JSON.parse(incoming) : incoming;
    if (!payload || !payload.beats) return;

    // A different run replaces the view outright and plays from the top;
    // more beats for the run already showing just extend it.
    const switched = !data || (payload.run && payload.run !== data.run);
    if (switched) {
      data = null;
      beat = 0;
    }

    const bank = root.querySelector('.asset-bank');
    Object.entries(payload.assets || {}).forEach(([key, uri]) => {
      if (bank.querySelector('img[data-key="' + key + '"]')) return;
      const picture = document.createElement('img');
      picture.dataset.key = key;
      picture.src = uri;
      picture.alt = '';
      bank.append(picture);
    });

    // Keep walking from wherever we are, so newly arrived rounds animate in
    // rather than the view snapping to the end of the run.
    const following = !data || beat >= data.beats.length - 1;
    data = payload;
    stopAt = null;
    open = null;
    if (following) { playing = !reduced(); schedule(); }
    draw();
  };

  // Gradio delivers rounds by swapping the markup of a separate element, which
  // leaves this viewer's DOM untouched. Watch for it rather than relying on a
  // Gradio event handler, which writes its return value back into the sender.
  let lastFeed = null;
  function readFeed(node) {
    const raw = node && node.getAttribute('data-vasa-feed');
    if (!raw || raw === lastFeed || !root) return;
    try {
      const bytes = Uint8Array.from(atob(raw), (c) => c.charCodeAt(0));
      window.vasaPush(JSON.parse(new TextDecoder().decode(bytes)));
      lastFeed = raw;          // only after it lands, so a failed read retries
    } catch (error) { /* a partial or malformed push is simply ignored */ }
  }
  const sweepFeeds = () => document.querySelectorAll('[data-vasa-feed]').forEach(readFeed);

  // Mount before wiring the feed: a push arriving first would find no viewer.
  root = host.querySelector('.vasa-report');
  if (root) load(read());

  const observer = new MutationObserver(() => {
    const found = host.querySelector('.vasa-report');
    if (found && found !== root) { root = found; load(read()); sweepFeeds(); }
  });
  observer.observe(host, {childList: true, subtree: true});

  const feedWatcher = new MutationObserver(sweepFeeds);
  feedWatcher.observe(document.body, {childList: true, subtree: true, attributes: true,
                                      attributeFilter: ['data-vasa-feed']});
  sweepFeeds();
}
