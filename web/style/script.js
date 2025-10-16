async function loadModel() {
    const model = await tf.loadLayersModel('model/model.json');
    return model;
}

const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

imageInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = async () => {
        const tensor = tf.browser.fromPixels(img, 1)
            .resizeNearestNeighbor([28, 28]) 
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims();

        const model = await loadModel();
        const prediction = model.predict(tensor);
        const predictedLabel = prediction.argMax(-1).dataSync()[0];
        const predictedLetter = String.fromCharCode(65 + predictedLabel);
        result.textContent = `Kết quả: ${predictedLetter}`;
        
        tensor.dispose();
        prediction.dispose();
    };
});