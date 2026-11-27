/* Hover and keyboard read-out for the progress chart. The chart itself is drawn on the server,
 * so the page reads and prints correctly without this script; it only adds the crosshair. */
(function () {
  'use strict';
  var source = document.getElementById('pv-data');
  var svg = document.querySelector('.pv-chart');
  if (!source || !svg) return;
  var data = JSON.parse(source.textContent);
  var points = data.points;
  if (!points.length) return;
  var plate = svg.querySelector('.pv-plate');
  var crosshair = svg.querySelector('.pv-crosshair');
  var dot = svg.querySelector('.pv-hover-dot');
  var tip = document.getElementById('pv-tooltip');
  var figure = tip.parentNode;
  var VIEW_WIDTH = svg.viewBox.baseVal.width;
  var shown = -1;

  function xOf(point) { return data.x0 + (data.x1 - data.x0) * (point.t - data.start) / (data.end - data.start); }
  function yOf(percent) { return data.y0 + (data.y100 - data.y0) * percent / 100; }

  function row(kind, value, label) {
    var line = document.createElement('div');
    line.className = 'pv-tip-row';
    var key = document.createElement('span');
    key.className = 'pv-key pv-key-' + kind;
    var strong = document.createElement('strong');
    strong.textContent = value;          // the value leads, the series name follows
    var name = document.createElement('span');
    name.textContent = label;
    line.appendChild(key);
    line.appendChild(strong);
    line.appendChild(name);
    return line;
  }

  function show(index) {
    index = Math.max(0, Math.min(points.length - 1, index));
    if (index === shown) return;
    shown = index;
    var point = points[index];
    var x = xOf(point);
    crosshair.setAttribute('x1', x);
    crosshair.setAttribute('x2', x);
    crosshair.setAttribute('visibility', 'visible');
    dot.setAttribute('cx', x);
    dot.setAttribute('cy', yOf(point.actual));
    dot.setAttribute('visibility', 'visible');
    tip.textContent = '';
    var heading = document.createElement('div');
    heading.className = 'pv-tip-date';
    heading.textContent = point.label;
    tip.appendChild(heading);
    tip.appendChild(row('actual', point.actual.toFixed(1) + '%', 'read so far'));
    if (point.even !== null) tip.appendChild(row('needed', point.even.toFixed(1) + '%', 'even pace'));
    tip.hidden = false;
    var scale = svg.getBoundingClientRect().width / VIEW_WIDTH;
    var left = x * scale + 14;
    if (left + tip.offsetWidth > figure.clientWidth) left = x * scale - tip.offsetWidth - 14;
    tip.style.left = Math.max(0, left) + 'px';
    tip.style.top = Math.max(0, yOf(point.actual) * scale - tip.offsetHeight - 10) + 'px';
  }

  function hide() {
    shown = -1;
    tip.hidden = true;
    crosshair.setAttribute('visibility', 'hidden');
    dot.setAttribute('visibility', 'hidden');
  }

  // The pointer only has to be near a day, never on the line: snap to the closest one.
  function nearest(clientX) {
    var box = svg.getBoundingClientRect();
    var x = (clientX - box.left) / box.width * VIEW_WIDTH;
    var best = 0;
    for (var i = 1; i < points.length; i++) {
      if (Math.abs(xOf(points[i]) - x) < Math.abs(xOf(points[best]) - x)) best = i;
    }
    return best;
  }

  plate.addEventListener('pointermove', function (event) { show(nearest(event.clientX)); });
  plate.addEventListener('pointerdown', function (event) { show(nearest(event.clientX)); });
  plate.addEventListener('pointerleave', function () { if (document.activeElement !== plate) hide(); });
  plate.addEventListener('focus', function () { show(points.length - 1); });
  plate.addEventListener('blur', hide);
  plate.addEventListener('keydown', function (event) {
    if (event.key === 'ArrowLeft') { show((shown < 0 ? points.length : shown) - 1); event.preventDefault(); }
    else if (event.key === 'ArrowRight') { show(shown + 1); event.preventDefault(); }
    else if (event.key === 'Home') { show(0); event.preventDefault(); }
    else if (event.key === 'End') { show(points.length - 1); event.preventDefault(); }
    else if (event.key === 'Escape') { hide(); }
  });
})();
