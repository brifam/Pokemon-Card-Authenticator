from flask import Flask, render_template, request, jsonify
from PIL import Image

from src.predict import predict_card

import io


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read uploaded image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))

        # Ask our inference module to analyze it
        prediction, confidence = predict_card(image)

        # Return result to frontend
        return render_template(
            'index.html',
            result=prediction,
            confidence=confidence
        )

    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify({'error': 'Unable to process image'}), 500


if __name__ == '__main__':
    app.run(debug=True)
