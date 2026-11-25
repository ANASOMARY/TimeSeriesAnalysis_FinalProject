(function () {
  'use strict';

  // Destructive actions ask first.
  document.addEventListener('submit', function (event) {
    var message = event.target.getAttribute('data-confirm');
    if (message && !window.confirm(message)) {
      event.preventDefault();
    }
  });

  // "Copy link" buttons.
  document.addEventListener('click', function (event) {
    var button = event.target.closest('[data-copy]');
    if (!button) return;
    var field = document.getElementById(button.getAttribute('data-copy'));
    if (!field) return;
    var done = function () {
      var original = button.textContent;
      button.textContent = 'Copied';
      setTimeout(function () { button.textContent = original; }, 1800);
    };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(field.value).then(done, function () { field.select(); });
    } else {
      field.select();
      try { document.execCommand('copy'); done(); } catch (err) { /* the text stays selected for a manual copy */ }
    }
  });

  // Upload form: list the chosen files; a title and page range only make sense for a single file.
  var files = document.getElementById('upload-files');
  if (files) {
    var list = document.getElementById('upload-list');
    var single = document.getElementById('single-fields');
    files.addEventListener('change', function () {
      list.textContent = '';
      Array.prototype.forEach.call(files.files, function (file) {
        var item = document.createElement('li');
        item.textContent = file.name + ' · ' + (file.size / 1048576).toFixed(1) + ' MB';
        list.appendChild(item);
      });
      var many = files.files.length > 1;
      ['title', 'page_range'].forEach(function (name) {
        var input = single.querySelector('[name="' + name + '"]');
        input.disabled = many;
        if (many) input.value = '';
      });
    });
    document.getElementById('upload-form').addEventListener('submit', function (event) {
      var button = event.target.querySelector('button.primary');
      button.disabled = true;
      button.textContent = 'Uploading and encrypting…';
    });
  }
})();
