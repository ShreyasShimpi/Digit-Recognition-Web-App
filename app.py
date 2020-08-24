from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
model = tf.keras.models.load_model('mymodel.hdf5')


@app.route("/")
def man():
    return render_template('home.html')


@app.route("/predict", methods=['POST'])
def home():
    data = request.files['file']

    base = os.path.dirname(__file__)
    file_path = os.path.join(base, 'uploads', secure_filename(data.filename))
    data.save(file_path)

    image = cv2.imread(file_path, 0)
    image = cv2.resize(image, (28, 28))
    image = np.array(image, dtype=float)
    image = (image/255) - 0.5
    image = image.reshape((-1, 784))
    prediction = model.predict(image)
    pred = np.argmax(prediction, axis=1)
    return render_template('next.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)
