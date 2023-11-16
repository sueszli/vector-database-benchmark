from ray import serve
import os
import tempfile
import numpy as np
from starlette.requests import Request
from typing import Dict
import tensorflow as tf
TRAINED_MODEL_PATH = os.path.join(tempfile.gettempdir(), 'mnist_model.h5')

def train_and_save_model():
    if False:
        return 10
    mnist = tf.keras.datasets.mnist
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    (x_train, x_test) = (x_train / 255.0, x_test / 255.0)
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10)])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1)
    model.evaluate(x_test, y_test, verbose=2)
    model.summary()
    model.save(TRAINED_MODEL_PATH)
if not os.path.exists(TRAINED_MODEL_PATH):
    train_and_save_model()

@serve.deployment
class TFMnistModel:

    def __init__(self, model_path: str):
        if False:
            while True:
                i = 10
        import tensorflow as tf
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)

    async def __call__(self, starlette_request: Request) -> Dict:
        input_array = np.array((await starlette_request.json())['array'])
        reshaped_array = input_array.reshape((1, 28, 28))
        prediction = self.model(reshaped_array)
        return {'prediction': prediction.numpy().tolist(), 'file': self.model_path}
mnist_model = TFMnistModel.bind(TRAINED_MODEL_PATH)