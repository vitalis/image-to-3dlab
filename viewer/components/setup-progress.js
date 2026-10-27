// The progress panel on Setup & Status: what this setup is doing right now, what it became
// of, and the state the panel is painted with.
//
// The title used to be written once, when the download was started, and never touched again
// — so a 16 GB fetch that finished perfectly said "Downloading" forever, beside a bar at 100%
// and a detail line reading "done · 16.8 GB on disk". Two copies of a ternary wrote that
// sentence, which is how it drifted out of sync with its own log.
//
// Pure, so it can be exercised under Node (tests/test_viewer_js_modules.py) against the
// shipped file rather than a copy of these rules.

import { MARK } from '../studio/pages.js';

// The endings, and the mark each one wears. Taken from the studio's Activity list, which
// already marks a finished thing this way: one convention, not two.
const ENDINGS = {
  done: { mark: MARK.done, verb: 'Installed' },
  error: { mark: MARK.error, verb: 'Setup failed' },
  cancelled: { mark: MARK.cancelled, verb: 'Setup cancelled' },
};

/** Where a run has got to: one of the three endings, or still going. */
export function panelState(event) {
  const phase = event && event.phase;
  return phase in ENDINGS ? phase : 'running';
}

/** The verb for a run that has not finished yet, from what we know when it is started.

 *  `build_present` decides "Setting up" over "Downloading" and stays right for the whole run:
 *  a route with no build yet compiles first, and a route that is built only fetches weights. */
export function setupVerb(backend, action) {
  if (action === 'rebuild') return 'Rebuilding';
  if (backend && !backend.build_present) return 'Setting up';
  if (backend && (backend.setup_fetches_weights
      || backend.action === 'download' || backend.action === 'resume')) {
    return 'Downloading';
  }
  return 'Building';
}

/** The panel's title: what it became, marked, once the run is over; what it is doing until then.

 *  `done` is the phase the server emits after the exit code is checked and the bytes are
 *  counted, so it is the only honest moment to call the backend installed. */
export function runTitle(backend, action, event) {
  const label = (backend && backend.label) || 'this backend';
  const ending = ENDINGS[panelState(event)];
  if (ending) return `${ending.mark} ${ending.verb} · ${label}`;
  return `${setupVerb(backend, action)} ${label}`;
}

const SETUP_STALL_SECONDS = 180;

/** Where to poll a setup run. Tool installs (Blender, gltfpack) have no status page. */
export function setupStatusUrl(payload = {}) {
  if (payload.status_url) return payload.status_url;
  if (payload.backend) return `/api/setup/${encodeURIComponent(payload.backend)}/status`;
  return null;
}

/** A download ends on done, error, or cancelled. A tool install ends on setup_done. */
export function isSetupFinished(status, event) {
  const phase = event?.phase;
  if (phase === 'done' || phase === 'error' || phase === 'cancelled' || phase === 'setup_done') return true;
  return status === 'done' || status === 'error' || status === 'cancelled';
}

/** Add one new log line, or replace the whole tail from a status poll. Do not glue the tail on twice. */
export function mergeSetupLog(shown, incoming, { replace = false } = {}) {
  const prior = shown || '';
  const next = String(incoming ?? '');
  if (!next) return prior;
  if (replace) return next;
  if (!prior || prior === next) return next || prior;
  if (!next.includes('\n') && (prior.endsWith(`\n${next}`) || prior === next)) return prior;
  return `${prior}\n${next}`;
}

/** Skip events a reconnected stream sends again. `index` starts at 0 on each connection. */
export function acceptStreamEvent(seenCount, index) {
  if (index < seenCount) return { accept: false, seenCount };
  return { accept: true, seenCount: index + 1 };
}

/**
 * Follow a setup run the way other jobs are polled.
 * The event stream is extra. If it drops, say so and keep checking until the run ends.
 * A tool install has no status page, so a dropped stream is reopened and old lines are skipped.
 */
export function openSetupWatch(payload, hooks = {}) {
  const fetchImpl = hooks.fetchImpl || fetch;
  const now = hooks.now || (() => Date.now());
  const EventSourceImpl = hooks.EventSourceImpl || globalThis.EventSource;
  const setTimer = hooks.setTimer || setTimeout;
  const clearTimer = hooks.clearTimer || clearTimeout;
  const interval = hooks.interval ?? 1200;
  const statusUrl = setupStatusUrl(payload);
  let stopped = false;
  let seen = 0;
  let source = null;
  let pollTimer = null;
  let lastChange = now();
  let lastKey = '';
  let log = '';

  const stop = () => {
    if (stopped) return;
    stopped = true;
    if (pollTimer != null) clearTimer(pollTimer);
    pollTimer = null;
    source?.close();
  };

  const note = (event, status, extra, replace) => {
    if (extra) {
      const merged = mergeSetupLog(log, extra, { replace });
      if (merged !== log) {
        log = merged;
        hooks.onLog?.(log);
      }
    }
    hooks.onEvent?.(event, status);
    if (isSetupFinished(status, event)) stop();
  };

  const poll = async () => {
    if (stopped || !statusUrl) return;
    try {
      const response = await fetchImpl(statusUrl);
      if (stopped) return;
      if (response.status === 404) {
        let message = 'This setup is no longer running.';
        try {
          const body = await response.json();
          if (body?.error) message = body.error;
        } catch { /* an old server answers with a file page, not JSON */ }
        note({ phase: 'error', message }, 'error', message, true);
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const body = await response.json();
      if (stopped) return;
      const event = body.last_event || {};
      const key = JSON.stringify(event) + (body.log_tail || '').length;
      if (key !== lastKey) {
        lastKey = key;
        lastChange = now();
      }
      note(event, body.status, body.log_tail || '', true);
      if (stopped) return;
    } catch {
      if (stopped) return;
      const stalled = (now() - lastChange) / 1000 > SETUP_STALL_SECONDS;
      hooks.onDrop?.('The connection dropped. Still checking…', { stalled });
    }
    if (stopped) return;
    pollTimer = setTimer(poll, interval);
  };

  const openStream = () => {
    if (stopped || !payload?.events_url || typeof EventSourceImpl !== 'function') return;
    source?.close();
    let index = 0;
    source = new EventSourceImpl(payload.events_url);
    source.onmessage = (message) => {
      if (stopped) return;
      let event;
      try { event = JSON.parse(message.data); } catch { return; }
      const decision = acceptStreamEvent(seen, index);
      index += 1;
      seen = decision.seenCount;
      if (!decision.accept) return;
      const line = event.log || (event.phase === 'setup' || event.phase === 'setup_done' ? event.message : '');
      note(event, event.status, line, false);
    };
    source.onerror = () => {
      if (stopped) return;
      hooks.onDrop?.('The connection dropped. Still checking…');
      index = 0;
      if (statusUrl) return;
      source.close();
      if (pollTimer != null) return;
      pollTimer = setTimer(() => {
        pollTimer = null;
        if (!stopped) openStream();
      }, interval);
    };
  };

  if (statusUrl) poll();
  openStream();
  return { stop };
}