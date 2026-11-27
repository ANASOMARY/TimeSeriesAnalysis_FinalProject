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
    var marks = document.createElement('div');
    marks.className = 'marks';
    el.appendChild(marks);
    pagesEl.appendChild(el);
    return { n: index + 1, w: size[0], h: size[1], el: el, canvas: canvas, marks: marks, near: false, loading: false, have: 0 };
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
            fetchOutlines(leaf);
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
        refreshTarget();
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

  // Reading progress. A page counts as read once at least half of it (or enough of it to fill
  // half the window) has been on screen, with the window in focus, for a few seconds in total.
  // Pages that were only preloaded, or scrolled past, do not count.
  var READ_SECONDS = 5;
  var onScreen = {};
  var countedAsRead = {};
  if (!cfg.isAuthor) {
    setInterval(function () {
      if (ended || document.hidden || !curtain.hidden) return;
      var frame = stage.getBoundingClientRect();
      var finished = [];
      leaves.forEach(function (leaf) {
        if (!leaf.have || countedAsRead[leaf.n]) return;
        var box = leaf.el.getBoundingClientRect();
        var visible = Math.min(box.bottom, frame.bottom) - Math.max(box.top, frame.top);
        if (visible < 0.5 * box.height && visible < 0.5 * frame.height) return;
        onScreen[leaf.n] = (onScreen[leaf.n] || 0) + 1;
        if (onScreen[leaf.n] >= READ_SECONDS) {
          countedAsRead[leaf.n] = true;
          finished.push(leaf.n);
        }
      });
      if (!finished.length) return;
      post(cfg.readUrl, { pages: finished }).then(function (response) {
        if (GONE.indexOf(response.status) !== -1) {
          return response.json().catch(function () { return {}; }).then(function (info) {
            endAccess(info.message || 'Your access to this chapter has ended.');
          });
        }
        if (!response.ok) throw new Error('not recorded');
      }).catch(function () {   // offline or a server hiccup: report these pages again on a later tick
        finished.forEach(function (n) { countedAsRead[n] = false; onScreen[n] = READ_SECONDS - 1; });
      });
    }, 1000);
  }

  // ------------------------------------------------- pinning notes to passages
  // The server sends paragraph outlines: rectangles only, never text. Clicking one, or dragging
  // a box anywhere on a page, selects the passage the next note will be pinned to.
  var outlines = {};       // page number -> list of [left, top, right, bottom] as fractions of the page
  var selection = null;    // { page, box }
  var drag = null;
  var afterDrag = false;
  var hoverMark = makeMark('hover');
  var pickMark = makeMark('pick');
  var noteScope = document.getElementById('note-scope');
  var noteClear = document.getElementById('note-clear');

  function makeMark(kind) {
    var el = document.createElement('div');
    el.className = 'mark ' + kind;
    return el;
  }

  function place(el, box) {
    el.style.left = (box[0] * 100) + '%';
    el.style.top = (box[1] * 100) + '%';
    el.style.width = ((box[2] - box[0]) * 100) + '%';
    el.style.height = ((box[3] - box[1]) * 100) + '%';
  }

  function fetchOutlines(leaf) {
    if (!cfg.canComment || outlines[leaf.n]) return;
    outlines[leaf.n] = [];
    fetch(cfg.blocksUrl + leaf.n, { credentials: 'same-origin', cache: 'no-store' })
      .then(function (response) { return response.ok ? response.json() : { blocks: [] }; })
      .then(function (data) { outlines[leaf.n] = data.blocks || []; })
      .catch(function () { delete outlines[leaf.n]; });
  }

  function leafOf(target) {
    var el = target && target.closest ? target.closest('.leaf') : null;
    return el ? leaves[Number(el.getAttribute('data-n')) - 1] : null;
  }

  function pointOn(leaf, event) {
    var rect = leaf.el.getBoundingClientRect();
    return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
            Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
  }

  function outlineAt(leaf, point) {
    var best = null;
    (outlines[leaf.n] || []).forEach(function (box) {
      if (point[0] >= box[0] && point[0] <= box[2] && point[1] >= box[1] && point[1] <= box[3]) {
        if (!best || (box[2] - box[0]) * (box[3] - box[1]) < (best[2] - best[0]) * (best[3] - best[1])) best = box;
      }
    });
    return best;
  }

  function refreshTarget() {
    if (!notePage) return;
    notePage.textContent = String(selection ? selection.page : current);
    noteScope.textContent = selection ? '· selected passage' : '· whole page';
    noteClear.hidden = !selection;
  }

  function select(leaf, box) {
    selection = { page: leaf.n, box: box };
    place(pickMark, box);
    leaf.marks.appendChild(pickMark);
    hoverMark.remove();
    refreshTarget();
    if (notesPanel.hidden) toggleNotes(true);
    document.getElementById('note-body').focus({ preventScroll: true });
  }

  function clearSelection() {
    selection = null;
    pickMark.remove();
    refreshTarget();
  }

  if (cfg.canComment) {
    document.body.classList.add('can-mark');
    pagesEl.addEventListener('pointermove', function (event) {
      if (ended) return;
      if (drag) {
        var now = pointOn(drag.leaf, event);
        if (!drag.moved && Math.abs(event.clientX - drag.x) + Math.abs(event.clientY - drag.y) > 8) drag.moved = true;
        if (drag.moved) {
          drag.box = [Math.min(drag.start[0], now[0]), Math.min(drag.start[1], now[1]),
                      Math.max(drag.start[0], now[0]), Math.max(drag.start[1], now[1])];
          place(pickMark, drag.box);
          drag.leaf.marks.appendChild(pickMark);
          hoverMark.remove();
        }
        return;
      }
      var leaf = leafOf(event.target);
      var box = leaf && event.pointerType !== 'touch' ? outlineAt(leaf, pointOn(leaf, event)) : null;
      if (box && !event.target.closest('.mark.pinned')) {
        place(hoverMark, box);
        leaf.marks.appendChild(hoverMark);
      } else {
        hoverMark.remove();
      }
    });
    pagesEl.addEventListener('pointerleave', function () { hoverMark.remove(); });
    pagesEl.addEventListener('pointerdown', function (event) {
      var leaf = leafOf(event.target);
      // Dragging a box is for mouse and pen; on touch screens a drag scrolls, and a tap selects a paragraph.
      if (!leaf || ended || event.button !== 0 || event.pointerType === 'touch' || event.target.closest('.mark.pinned')) return;
      drag = { leaf: leaf, start: pointOn(leaf, event), x: event.clientX, y: event.clientY, moved: false, box: null };
    });
    window.addEventListener('pointerup', function () {
      if (!drag) return;
      var done = drag;
      drag = null;
      if (!done.moved) return;  // a plain click, handled by the click listener
      afterDrag = true;
      setTimeout(function () { afterDrag = false; }, 0);
      if (done.box && done.box[2] - done.box[0] > 0.02 && done.box[3] - done.box[1] > 0.008) {
        select(done.leaf, done.box);
      } else if (selection) {
        place(pickMark, selection.box);
        leaves[selection.page - 1].marks.appendChild(pickMark);
      } else {
        pickMark.remove();
      }
    });
    pagesEl.addEventListener('click', function (event) {
      if (afterDrag || ended || event.target.closest('.mark.pinned')) return;
      var leaf = leafOf(event.target);
      if (!leaf) return;
      var box = outlineAt(leaf, pointOn(leaf, event));
      if (box) select(leaf, box);
      else if (selection) clearSelection();
    });
    noteClear.addEventListener('click', clearSelection);
  }

  function flash(el) {
    if (!el) return;
    el.classList.remove('flash');
    void el.offsetWidth;  // restart the animation
    el.classList.add('flash');
  }

  function drawNoteMarks() {
    leaves.forEach(function (leaf) {
      Array.prototype.slice.call(leaf.marks.querySelectorAll('.mark.pinned')).forEach(function (el) { el.remove(); });
    });
    notes.forEach(function (note) {
      var leaf = leaves[note.page - 1];
      if (!note.box || !leaf) return;
      var el = makeMark('pinned');
      el.setAttribute('data-id', String(note.id));
      var kinds = (note.categories || []).map(function (category) { return category.name; }).join(', ');
      el.title = (cfg.isAuthor ? note.author + ': ' : '') + [kinds, note.body].filter(Boolean).join(' \u2014 ');
      if (note.categories && note.categories.length) el.classList.add('cat', 'c-' + note.categories[0].color);
      place(el, note.box);
      el.addEventListener('click', function (event) { event.stopPropagation(); showNote(note.id); });
      leaf.marks.insertBefore(el, leaf.marks.firstChild);
    });
  }

  // From a highlight on the page to its note in the panel ...
  function showNote(id) {
    if (notesPanel.hidden) toggleNotes(true);
    var item = noteList.querySelector('.note[data-id="' + id + '"]');
    if (item) {
      item.scrollIntoView({ block: 'nearest' });
      flash(item);
    }
  }

  // ... and from a note in the panel to its place on the page.
  function showMark(note) {
    var leaf = leaves[note.page - 1];
    if (!leaf) return;
    if (!note.box) { goTo(note.page); return; }
    stage.scrollTop = leaf.el.offsetTop + note.box[1] * leaf.el.offsetHeight - stage.clientHeight * 0.3;
    flash(leaf.marks.querySelector('.mark.pinned[data-id="' + note.id + '"]'));
  }

  // The author's "Show on page" buttons open the reader at #p=<page>&n=<note>.
  function openFromHash() {
    var match = /p=(\d+)(?:&n=(\d+))?/.exec(window.location.hash);
    if (!match) return;
    var wanted = match[2] ? notes.filter(function (note) { return note.id === Number(match[2]); })[0] : null;
    if (wanted) {
      toggleNotes(true);
      showMark(wanted);
      showNote(wanted.id);
    } else {
      goTo(Number(match[1]));
    }
  }

  // ----------------------------------------------------------- categories
  // The author defines the kinds of note. A reviewer taps any that apply; the line under the
  // buttons explains whichever one is pointed at, focused or was picked last, so the meaning is
  // visible on touch screens too, where there is no hover.
  var categories = cfg.categories || [];

  function makePicker(selectedIds, alreadyOnNote) {
    var options = categories.slice();
    (alreadyOnNote || []).forEach(function (category) {   // a retired category stays on a note that has it
      if (!options.some(function (other) { return other.id === category.id; })) options.push(category);
    });
    var chosen = (selectedIds || []).filter(function (id) {
      return options.some(function (other) { return other.id === id; });
    });
    var wrap = document.createElement('div');
    wrap.className = 'catpick';
    var picker = { el: wrap, get: function () { return chosen.slice(); }, reset: function () {} };
    if (!options.length) { wrap.hidden = true; return picker; }

    var caption = document.createElement('div');
    caption.className = 'cat-caption';
    caption.textContent = 'Kind of note (optional, pick any that apply)';
    var row = document.createElement('div');
    row.className = 'chips';
    row.setAttribute('role', 'group');
    row.setAttribute('aria-label', 'Kind of note');
    var help = document.createElement('p');
    help.className = 'cat-help';
    help.setAttribute('aria-live', 'polite');

    function describe(category) { return category.name + ': ' + (category.description || 'No explanation given.'); }
    function resting() {
      if (!chosen.length) return 'Point at or tap a category to see what it means.';
      var last = chosen[chosen.length - 1];
      return describe(options.filter(function (other) { return other.id === last; })[0]);
    }
    function explain(category) { help.textContent = category ? describe(category) : resting(); }

    var buttons = options.map(function (category) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'chip c-' + category.color;
      button.textContent = category.name;
      button.setAttribute('aria-pressed', String(chosen.indexOf(category.id) !== -1));
      button.addEventListener('click', function () {
        var at = chosen.indexOf(category.id);
        if (at === -1) chosen.push(category.id); else chosen.splice(at, 1);
        button.setAttribute('aria-pressed', String(at === -1));
        explain(at === -1 ? category : null);
      });
      button.addEventListener('mouseenter', function () { explain(category); });
      button.addEventListener('focus', function () { explain(category); });
      button.addEventListener('mouseleave', function () { explain(null); });
      button.addEventListener('blur', function () { explain(null); });
      row.appendChild(button);
      return button;
    });
    picker.reset = function () {
      chosen = [];
      buttons.forEach(function (button) { button.setAttribute('aria-pressed', 'false'); });
      explain(null);
    };
    explain(null);
    wrap.appendChild(caption);
    wrap.appendChild(row);
    wrap.appendChild(help);
    return picker;
  }

  function categoryLabels(list) {
    var tags = document.createElement('div');
    tags.className = 'note-tags';
    list.forEach(function (category) {
      var tag = document.createElement('span');
      tag.className = 'cat-label c-' + category.color;
      tag.textContent = category.name;
      tag.title = category.description || '';
      tags.appendChild(tag);
    });
    return tags;
  }

  var notePicker = null;
  if (noteForm) {
    notePicker = makePicker([]);
    document.getElementById('note-cats').appendChild(notePicker.el);
    if (categories.length) {
      document.getElementById('note-body').placeholder = 'Add details (optional if you picked a category)\u2026';
    }
  }

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
      drawNoteMarks();
      return;
    }
    var order = function (a, b) {
      return a.page - b.page || (a.box ? a.box[1] : -1) - (b.box ? b.box[1] : -1) || a.id - b.id;
    };
    notes.slice().sort(order).forEach(function (note) {
      var item = document.createElement('div');
      item.className = 'note';
      item.setAttribute('data-page', String(note.page));
      item.setAttribute('data-id', String(note.id));
      var meta = document.createElement('div');
      meta.className = 'note-meta';
      var jump = document.createElement('button');
      jump.type = 'button';
      jump.textContent = 'Page ' + note.page + (note.box ? ' · passage' : '') + (cfg.isAuthor ? ' · ' + note.author : '');
      jump.addEventListener('click', function () { showMark(note); });
      meta.appendChild(jump);
      var body = document.createElement('div');
      body.className = 'note-body';
      body.textContent = note.body;
      body.hidden = !note.body;
      if (note.mine && cfg.canComment) {
        var actions = document.createElement('span');
        actions.className = 'note-actions';
        var edit = document.createElement('button');
        edit.type = 'button';
        edit.textContent = 'Edit';
        edit.addEventListener('click', function () { editNote(note, item, body); });
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
        actions.appendChild(edit);
        actions.appendChild(del);
        meta.appendChild(actions);
      }
      item.appendChild(meta);
      if (note.categories && note.categories.length) item.appendChild(categoryLabels(note.categories));
      if (note.quote) {
        var quote = document.createElement('div');
        quote.className = 'note-quote';
        quote.textContent = '\u201c' + (note.quote.length > 260 ? note.quote.slice(0, 260) + '\u2026' : note.quote) + '\u201d';
        item.appendChild(quote);
      }
      item.appendChild(body);
      if (note.edited) {
        var edited = document.createElement('div');
        edited.className = 'note-edited';
        edited.textContent = 'Edited';
        item.appendChild(edited);
      }
      noteList.appendChild(item);
    });
    markNotes();
    drawNoteMarks();
  }

  // Swap a note's text for an editing box, in place. The passage it is pinned to does not move.
  function editNote(note, item, body) {
    if (item.querySelector('.note-edit')) return;
    var form = document.createElement('form');
    form.className = 'note-edit';
    var picker = makePicker((note.categories || []).map(function (category) { return category.id; }), note.categories);
    var field = document.createElement('textarea');
    field.maxLength = 5000;
    field.rows = Math.min(10, Math.max(3, Math.ceil(note.body.length / 38)));
    field.value = note.body;
    field.setAttribute('aria-label', 'Edit note on page ' + note.page);
    var save = document.createElement('button');
    save.className = 'vbtn solid';
    save.textContent = 'Save';
    var cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.className = 'vbtn';
    cancel.textContent = 'Cancel';
    cancel.addEventListener('click', renderNotes);
    field.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') { event.stopPropagation(); renderNotes(); }
    });
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var text = field.value.trim();
      var kinds = picker.get();
      if (!text && !kinds.length) { say('A note needs a category or some text. Delete it if you no longer want it.'); return; }
      save.disabled = true;
      post(cfg.commentsUrl + '/' + note.id + '/edit', { body: text, categories: kinds }).then(function (response) {
        return response.json().then(function (data) {
          if (response.ok) {
            notes = notes.map(function (other) { return other.id === note.id ? data : other; });
            renderNotes();
          } else if (response.status === 403 && data.error && data.error !== 'Forbidden') {
            endAccess(data.message);
          } else {
            save.disabled = false;
            say(data.message || 'The note could not be saved.');
          }
        });
      }).catch(function () { save.disabled = false; say('The note could not be saved. Check your connection.'); });
    });
    form.appendChild(picker.el);
    form.appendChild(field);
    form.appendChild(save);
    form.appendChild(cancel);
    var shown = item.querySelector('.note-tags');   // the editor shows the categories itself
    if (shown) shown.remove();
    item.replaceChild(form, body);
    field.focus();
    field.setSelectionRange(field.value.length, field.value.length);
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
      var kinds = notePicker.get();
      if (!text && !kinds.length) {
        say(categories.length ? 'Pick a category or write a note.' : 'Write a note first.');
        return;
      }
      var note = selection ? { page: selection.page, body: text, box: selection.box } : { page: current, body: text };
      note.categories = kinds;
      post(cfg.commentsUrl, note).then(function (response) {
        return response.json().then(function (data) {
          if (response.ok) {
            notes.push(data);
            field.value = '';
            notePicker.reset();
            clearSelection();
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
    .then(function (list) { notes = list; renderNotes(); openFromHash(); })
    .catch(function () {});

  // ----------------------------------------------------------------- start
  layout();
  leaves.forEach(function (leaf) { observer.observe(leaf.el); });
  stage.focus();
})();
