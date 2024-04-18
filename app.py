from flask import Flask, render_template, request, jsonify
import os
    
# from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/gesture_images'


# Load TinyBERT model and tokenizer
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict_text', methods=['GET','POST'])
def predict_sentiment():
    if request.method == 'POST':
        data = request.json
        text = data['text']

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Process model outputs
        sentiment_scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
        # Assuming binary sentiment classification (positive vs. negative)
        positive_score = sentiment_scores[1]  # Probability of being positive
        negative_score = sentiment_scores[0]  # Probability of being negative

        # Determine sentiment label based on scores
        sentiment_label = 'Positive' if positive_score > negative_score else 'Negative'

        return jsonify({'sentiment': sentiment_label, 'positive_score': positive_score, 'negative_score': negative_score})
    else:
        return jsonify({'error': 'Method not allowed. Please use POST method.'})

@app.route('/tinybert_form')
def tinybert_form():
    # Render the TinyBERT form page template
    return render_template('tinybert.html')
    
@app.route('/upload_sign', methods=['GET', 'POST'])
def upload_sign():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
    
    return render_template('upload.html')  # Display the form for uploading signatures



@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Perform sign language prediction here
    prediction = predict_sign_language(image_path)

    return render_template('result.html', prediction=prediction)


def predict_sign_language(image_path):
    filename = os.path.basename(image_path)
    sign = filename.split('.')[0]

    return sign


if __name__ == '__main__':
    app.run(debug=True)
