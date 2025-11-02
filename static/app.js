// Preview ảnh chọn + drag & drop + nút loading
(function () {
  const fileInput = document.getElementById('image');
  const drop = document.getElementById('drop');
  const previewImg = document.getElementById('preview-img');
  const meta = document.getElementById('meta');
  const selectBtn = document.getElementById('select-btn');
  const submitBtn = document.getElementById('submit-btn');
  const form = document.getElementById('form');

  function preview(file) {
    if (!file) return;
    if (!file.type.match(/^image\//)) {
      meta.textContent = 'Vui lòng chọn tệp ảnh (PNG/JPG).';
      previewImg.src = '';
      return;
    }
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    meta.textContent =
      `Tên: ${file.name}\nKích thước: ${(file.size/1024).toFixed(1)} KB\nLoại: ${file.type}`;
    submitBtn.disabled = false;
  }

  selectBtn?.addEventListener('click', () => fileInput?.click());
  fileInput?.addEventListener('change', (e) => preview(e.target.files?.[0]));

  ;['dragenter','dragover'].forEach(evt => {
    drop?.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.add('dragover'); }, false);
  });
  ;['dragleave','drop'].forEach(evt => {
    drop?.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); drop.classList.remove('dragover'); }, false);
  });
  drop?.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const file = dt?.files?.[0];
    if (file) {
      // đồng bộ với input để form submit được
      const dT = new DataTransfer();
      dT.items.add(file);
      fileInput.files = dT.files;
      preview(file);
    }
  });

  // submit: show loading
  form?.addEventListener('submit', () => {
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span>Đang nhận diện...';
  });
})();
