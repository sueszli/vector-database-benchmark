accuracy = 42
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
import keras

def objective(config):
    if False:
        for i in range(10):
            print('nop')
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(784, activation=config['activation']))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return {'accuracy': accuracy}
search_space = {'activation': tune.choice(['relu', 'tanh'])}
algo = HyperOptSearch()
tuner = tune.Tuner(objective, tune_config=tune.TuneConfig(metric='accuracy', mode='max', search_alg=algo), param_space=search_space)
results = tuner.fit()