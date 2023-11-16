from ray import serve
import pickle
import json
import numpy as np
import os
import tempfile
from starlette.requests import Request
from typing import Dict
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
model = GradientBoostingClassifier()
iris_dataset = load_iris()
(data, target, target_names) = (iris_dataset['data'], iris_dataset['target'], iris_dataset['target_names'])
(np.random.shuffle(data), np.random.shuffle(target))
(train_x, train_y) = (data[:100], target[:100])
(val_x, val_y) = (data[100:], target[100:])
model.fit(train_x, train_y)
print('MSE:', mean_squared_error(model.predict(val_x), val_y))
MODEL_PATH = os.path.join(tempfile.gettempdir(), 'iris_model_gradient_boosting_classifier.pkl')
LABEL_PATH = os.path.join(tempfile.gettempdir(), 'iris_labels.json')
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(LABEL_PATH, 'w') as f:
    json.dump(target_names.tolist(), f)

@serve.deployment
class BoostingModel:

    def __init__(self, model_path: str, label_path: str):
        if False:
            i = 10
            return i + 15
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(label_path) as f:
            self.label_list = json.load(f)

    async def __call__(self, starlette_request: Request) -> Dict:
        payload = await starlette_request.json()
        print('Worker: received starlette request with data', payload)
        input_vector = [payload['sepal length'], payload['sepal width'], payload['petal length'], payload['petal width']]
        prediction = self.model.predict([input_vector])[0]
        human_name = self.label_list[prediction]
        return {'result': human_name}
boosting_model = BoostingModel.bind(MODEL_PATH, LABEL_PATH)