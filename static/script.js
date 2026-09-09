const navToggle = document.getElementById('nav-toggle');
const mainNav = document.getElementById('main-nav');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('card-file-input');
const uploadForm = document.getElementById('upload-form');


navToggle.addEventListener("click", () => {
    const isOpen = mainNav.classList.toggle("nav-open");
    navToggle.setAttribute("aria-expanded", isOpen);
});


['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    window.addEventListener(eventName, (e) => e.preventDefault(), false);
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});


['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});


function handleFileUpload(files) {
    if (files.length > 0) {
        fileInput.files = files; 
        console.log("File attached:", files[0].name);
        dropZone.querySelector('h2').textContent = `Analyzing: ${files[0].name}`;
        
        uploadForm.submit();
    }
}


dropZone.addEventListener('drop', (e) => {
    const droppedFiles = e.dataTransfer.files;
    handleFileUpload(droppedFiles);
});


fileInput.addEventListener('change', (e) => {
    handleFileUpload(e.target.files);
});