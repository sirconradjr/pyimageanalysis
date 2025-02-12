from flask import Flask, render_template, request, jsonify, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from mtcnn.mtcnn import MTCNN
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100

classes = []
data = []
labels = []
model = None

def preprocess_image(image_path):
    detector = MTCNN()
    image = cv2.imread(image_path)
    if image is None:
        return None
    faces = detector.detect_faces(image)
    if not faces:
        return None
    x, y, width, height = faces[0]['box']
    face = image[y:y+height, x:x+width]
    face = cv2.resize(face, IMAGE_SIZE)
    return np.array(face) / 255.0

def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

@app.route('/')
def index():
    return render_template('index.html', classes=classes)

@app.route('/upload', methods=['POST'])
def upload():
    global classes, data, labels
    class_name = request.form.get('class_name')

    if not class_name:
        return jsonify({'error': 'Class name is required'}), 400

    if class_name not in classes:
        classes.append(class_name)

    for file in request.files.getlist('images'):
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            processed_image = preprocess_image(filepath)
            if processed_image is not None:
                data.append(processed_image)
                labels.append(classes.index(class_name))

    return jsonify({'message': f'Uploaded images for class {class_name}'}), 200

@app.route('/train', methods=['POST'])
def train():
    global model

    if not data or not labels:
        return jsonify({'error': 'No images uploaded for training'}), 400

    X_train, X_val, y_train, y_val = train_test_split(np.array(data), np.array(labels), test_size=0.2, random_state=42)

    model = create_model(num_classes=len(classes))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Simulate progress for SweetAlert2
    for i in range(1, 101, 10):
        print(f"Training progress: {i}%")  # Simulate progress output

    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

    return jsonify({'message': 'Training Complete'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return jsonify({'error': 'Model is not trained yet'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    processed_image = preprocess_image(filepath)

    if processed_image is None:
        return jsonify({'error': 'No face detected in the image'}), 400

    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    predicted_class = classes[np.argmax(prediction)]

    return jsonify({
        'prediction': predicted_class,
        'image_url': url_for('static', filename=f'uploads/{filename}', _external=True)
    }), 200

if __name__ == '__main__':
    app.run()