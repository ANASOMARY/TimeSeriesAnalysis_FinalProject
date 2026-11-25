/* ChapterVault secure reader.
 *
 * Pages arrive as watermarked images rendered on the server; the manuscript file itself never
 * reaches the browser. This script draws them onto canvases, keeps only the pages near the
 * viewport in memory, and hides the content whenever the window loses focus or access ends.
 */
(function () {
  'use strict';

  var cfg = JSON.parse(document.getElementById('viewer-config').textContent);
  var stage = document.getElementById('stage');
  var pagesEl = document.getElementById('pages');
  var curtain = document.getElementById('curtain');
  var pageInput = document.getElementById('page-input');
  var zoomLabel = document.getElementById('zoom-label');
  var toast = document.getElementById('toast');
  var notesPanel = document.getElementById('notes');
  var notesToggle = document.getElementById('notes-toggle');
  var noteList = document.getElementById('note-list');
  var noteForm = document.getElementById('note-form');
  var notePage = document.getElementById('note-page');
  var notesCount = document.getElementById('notes-count');

  var BASE_WIDTH = 880;
  var MAX_ACTIVE = 3;
  var GONE = [401, 403, 404];  // signed out, access withdrawn, or chapter replaced/deleted
  var zoom = 1;
  var current = 1;
  var ended = false;
  var active = 0;
  var notes = [];
  var widest = Math.max.apply(null, cfg.pages.map(function (p) { return p[0]; }));

  var leaves = cfg.pages.map(function (size, index) {
    var el = document.createElement('div');
    el.className = 'leaf';
    el.setAttribute('data-n', String(index + 1));
    var canvas = document.createElement('canvas');
    canvas.width = 0;
    canvas.height = 0;
    el.appendChild(canvas);
    pagesEl.appendChild(el);
    return { n: index + 1, w: size[0], h: size[1], el: el, canvas: canvas, near: false, loading: false, have: 0 };
  });

  // ------------------------------------------------------------- layout
  function layout() {
    var fit = Math.min(BASE_WIDTH, stage.clientWidth - 32);
    leaves.forEach(function (leaf) {
      var width = Math.max(160, Math.round(fit * zoom * (leaf.w / widest)));
      leaf.el.style.width = width + 'px';
      leaf.el.style.height = Math.round(width * leaf.h / leaf.w) + 'px';
    });
    zoomLabel.textContent = Math.round(zoom * 100) + '%';
  }

  function setZoom(next) {
    next = Math.max(0.5, Math.min(2.5, next));
    if (next === zoom) return;
    var anchor = leaves[current - 1];
    var offset = (stage.scrollTop - anchor.el.offsetTop) / anchor.el.offsetHeight;
    zoom = next;
    layout();
    stage.scrollTop = anchor.el.offsetTop + offset * anchor.el.offsetHeight;
    pump();
  }

  // ------------------------------------------------------------ loading
  function wantedWidth(leaf) {
    var ratio = Math.min(window.devicePixelRatio || 1, 2);
    var width = Math.round(leaf.el.clientWidth * ratio / 200) * 200;
    return Math.max(400, Math.min(2200, width));
  }

  function unload(leaf) {
    leaf.canvas.width = 0;
    leaf.canvas.height = 0;
    leaf.have = 0;
    leaf.el.classList.remove('loaded');
  }

  function pump() {
    if (ended) return;
    for (var i = 0; i < leaves.length && active < MAX_ACTIVE; i++) {
      var leaf = leaves[i];
      if (leaf.near && !leaf.loading && leaf.have < wantedWidth(leaf)) {
        load(leaf);
      }
    }
  }

  function load(leaf) {
    var want = wantedWidth(leaf);
    leaf.loading = true;
    active++;
    fetch(cfg.pageUrl + leaf.n + '?w=' + want, { credentials: 'same-origin', cache: 'no-store' })
      .then(function (response) {
        if (response.status === 429) {
          return new Promise(function (resolve) { setTimeout(resolve, 5000); });
        }
        if (!response.ok) {
          return response.json().catch(function () { return {}; }).then(function (info) {
            if (GONE.indexOf(response.status) !== -1) {
              endAccess(info.message || 'Your access to this chapter has ended.');
            }
          });
        }
        return response.blob().then(createImageBitmap).then(function (bitmap) {
          if (leaf.near && !ended) {
            leaf.canvas.width = bitmap.width;
            leaf.canvas.height = bitmap.height;
            leaf.canvas.getContext('2d').drawImage(bitmap, 0, 0);
            leaf.have = want;
            leaf.el.classList.add('loaded');
          }
          bitmap.close();
        });
      })
      .catch(function () { /* offline or interrupted: retried on the next scroll or heartbeat */ })
      .then(function () {
        leaf.loading = false;
        active--;
        pump();
      });
  }

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      var leaf = leaves[Number(entry.target.getAttribute('data-n')) - 1];
      leaf.near = entry.isIntersecting;
      if (!leaf.near) unload(leaf);
    });
    pump();
  }, { root: stage, rootMargin: '900px 0px' });

  // ------------------------------------------------------- page tracking
  function findCurrent() {
    var line = stage.scrollTop + stage.clientHeight * 0.35;
    var low = 0, high = leaves.length - 1;
    while (low < high) {
      var mid = (low + high + 1) >> 1;
      if (leaves[mid].el.offsetTop <= line) low = mid; else high = mid - 1;
    }
    return low + 1;
  }

  var ticking = false;
  stage.addEventListener('scroll', function () {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      var page = findCurrent();
      if (page !== current) {
        current = page;
        if (document.activeElement !== pageInput) pageInput.value = String(page);
        if (notePage) notePage.textContent = String(page);
        markNotes();
      }
    });
  });

  function goTo(page) {
    page = Math.max(1, Math.min(leaves.length, page | 0));
    stage.scrollTop = leaves[page - 1].el.offsetTop - 10;
  }

  pageInput.addEventListener('change', function () { goTo(Number(pageInput.value)); stage.focus(); });
  pageInput.addEventListener('keydown', function (event) { if (event.key === 'Enter') { pageInput.blur(); } });
  document.getElementById('zoom-in').addEventListener('click', function () { setZoom(zoom + 0.25); });
  document.getElementById('zoom-out').addEventListener('click', function () { setZoom(zoom - 0.25); });
  window.addEventListener('resize', function () { layout(); pump(); });

  // ---------------------------------------------------------- protection
  function post(url, body) {
    return fetch(url, {
      method: 'POST', credentials: 'same-origin', cache: 'no-store',
      headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': cfg.csrf },
      body: JSON.stringify(body || {})
    });
  }

  function report(kind, detail) {
    post(cfg.eventUrl, { kind: kind, detail: detail || '', page: current }).catch(function () {});
  }

  function say(message) {
    toast.textContent = message;
    toast.hidden = false;
    clearTimeout(say.timer);
    say.timer = setTimeout(function () { toast.hidden = true; }, 3500);
  }

  function hide() { curtain.hidden = false; }
  function reveal() { if (!ended && !document.hidden) curtain.hidden = true; }

  function endAccess(message) {
    if (ended) return;
    ended = true;
    leaves.forEach(unload);
    observer.disconnect();
    document.getElementById('curtain-title').textContent = 'This chapter is closed';
    document.getElementById('curtain-text').textContent = message;
    document.getElementById('curtain-exit').hidden = false;
    curtain.hidden = false;
  }

  function editable(target) {
    return target && target.closest && target.closest('textarea, input');
  }

  window.addEventListener('blur', hide);
  window.addEventListener('focus', reveal);
  document.addEventListener('visibilitychange', function () { if (document.hidden) hide(); else reveal(); });
  curtain.addEventListener('click', function () { window.focus(); reveal(); });

  document.addEventListener('contextmenu', function (event) { if (!editable(event.target)) event.preventDefault(); });
  document.addEventListener('dragstart', function (event) { event.preventDefault(); });
  document.addEventListener('selectstart', function (event) { if (!editable(event.target)) event.preventDefault(); });
  document.addEventListener('copy', function (event) {
    if (!editable(event.target)) { event.preventDefault(); report('copy_attempt', 'copy'); }
  });

  document.addEventListener('keydown', function (event) {
    var key = (event.key || '').toLowerCase();
    var mod = event.ctrlKey || event.metaKey;
    if (mod && key === 'p') {
      event.preventDefault();
      report('print_attempt', 'keyboard shortcut');
      say('Printing is disabled for this review copy.');
    } else if (mod && key === 's') {
      event.preventDefault();
      report('copy_attempt', 'save shortcut');
      say('Saving is disabled for this review copy.');
    } else if (key === 'printscreen' || (event.metaKey && event.shiftKey && (key === '3' || key === '4' || key === '5' || key === 's'))) {
      hide();
      report('capture_key', event.key);
      setTimeout(reveal, 2500);
    } else if (!editable(event.target) && !mod) {
      if (key === '+' || key === '=') setZoom(zoom + 0.25);
      else if (key === '-') setZoom(zoom - 0.25);
    }
  });
  document.addEventListener('keyup', function (event) {
    if ((event.key || '').toLowerCase() === 'printscreen') {
      hide();
      report('capture_key', 'PrintScreen');
      setTimeout(reveal, 2500);
    }
  });
  window.addEventListener('beforeprint', function () { hide(); report('print_attempt', 'print dialog'); });
  window.addEventListener('afterprint', reveal);

  // Heartbeat: records reading time and closes the reader within seconds if access is withdrawn.
  setInterval(function () {
    if (ended || document.hidden) return;
    post(cfg.pingUrl, { page: current }).then(function (response) {
      if (GONE.indexOf(response.status) !== -1) {
        response.json().catch(function () { return {}; }).then(function (info) {
          endAccess(info.message || 'Your access to this chapter has ended.');
        });
      } else {
        pump();
      }
    }).catch(function () {});
  }, 20000);

  // ---------------------------------------------------------------- notes
  function markNotes() {
    Array.prototype.forEach.call(noteList.children, function (node) {
      node.classList.toggle('here', Number(node.getAttribute('data-page')) === current);
    });
  }

  function renderNotes() {
    noteList.textContent = '';
    notesCount.hidden = notes.length === 0;
    notesCount.textContent = String(notes.length);
    if (!notes.length) {
      var empty = document.createElement('p');
      empty.className = 'notes-empty';
      empty.textContent = cfg.isAuthor ? 'No reviewer has left feedback on this chapter yet.'
        : 'No notes yet. Notes are private between you and the author.';
      noteList.appendChild(empty);
      return;
    }
    notes.slice().sort(function (a, b) { return a.page - b.page || a.id - b.id; }).forEach(function (note) {
      var item = document.createElement('div');
      item.className = 'note';
      item.setAttribute('data-page', String(note.page));
      var meta = document.createElement('div');
      meta.className = 'note-meta';
      var jump = document.createElement('button');
      jump.type = 'button';
      jump.textContent = 'Page ' + note.page + (cfg.isAuthor ? ' · ' + note.author : '');
      jump.addEventListener('click', function () { goTo(note.page); });
      meta.appendChild(jump);
      if (note.mine) {
        var del = document.createElement('button');
        del.type = 'button';
        del.className = 'del';
        del.textContent = 'Delete';
        del.addEventListener('click', function () {
          if (!window.confirm('Delete this note?')) return;
          post(cfg.commentsUrl + '/' + note.id + '/delete').then(function (response) {
            if (response.ok) {
              notes = notes.filter(function (other) { return other.id !== note.id; });
              renderNotes();
            }
          });
        });
        meta.appendChild(del);
      }
      var body = document.createElement('div');
      body.className = 'note-body';
      body.textContent = note.body;
      item.appendChild(meta);
      item.appendChild(body);
      noteList.appendChild(item);
    });
    markNotes();
  }

  function toggleNotes(open) {
    notesPanel.hidden = !open;
    notesToggle.setAttribute('aria-expanded', String(open));
    layout();
    pump();
  }
  notesToggle.addEventListener('click', function () { toggleNotes(notesPanel.hidden); });
  document.getElementById('notes-close').addEventListener('click', function () { toggleNotes(false); });

  if (noteForm) {
    noteForm.addEventListener('submit', function (event) {
      event.preventDefault();
      var field = document.getElementById('note-body');
      var text = field.value.trim();
      if (!text) return;
      post(cfg.commentsUrl, { page: current, body: text }).then(function (response) {
        return response.json().then(function (data) {
          if (response.ok) {
            notes.push(data);
            field.value = '';
            renderNotes();
          } else if (response.status === 403 && data.error) {
            endAccess(data.message);
          } else {
            say(data.message || 'The note could not be saved.');
          }
        });
      }).catch(function () { say('The note could not be saved. Check your connection.'); });
    });
  }

  fetch(cfg.commentsUrl, { credentials: 'same-origin', cache: 'no-store' })
    .then(function (response) { return response.ok ? response.json() : []; })
    .then(function (list) { notes = list; renderNotes(); })
    .catch(function () {});

  // ----------------------------------------------------------------- start
  layout();
  leaves.forEach(function (leaf) { observer.observe(leaf.el); });
  stage.focus();
})();
