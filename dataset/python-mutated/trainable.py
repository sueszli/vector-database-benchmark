def objective(x, a, b):
    if False:
        i = 10
        return i + 15
    return a * x ** 0.5 + b
from ray import train, tune

def trainable(config: dict):
    if False:
        i = 10
        return i + 15
    intermediate_score = 0
    for x in range(20):
        intermediate_score = objective(x, config['a'], config['b'])
        train.report({'score': intermediate_score})
tuner = tune.Tuner(trainable, param_space={'a': 2, 'b': 4})
results = tuner.fit()
from ray import train, tune

def trainable(config: dict):
    if False:
        i = 10
        return i + 15
    final_score = 0
    for x in range(20):
        final_score = objective(x, config['a'], config['b'])
    train.report({'score': final_score})
tuner = tune.Tuner(trainable, param_space={'a': 2, 'b': 4})
results = tuner.fit()

def trainable(config: dict):
    if False:
        while True:
            i = 10
    final_score = 0
    for x in range(20):
        final_score = objective(x, config['a'], config['b'])
    return {'score': final_score}
from ray import train, tune

class Trainable(tune.Trainable):

    def setup(self, config: dict):
        if False:
            i = 10
            return i + 15
        self.x = 0
        self.a = config['a']
        self.b = config['b']

    def step(self):
        if False:
            for i in range(10):
                print('nop')
        score = objective(self.x, self.a, self.b)
        self.x += 1
        return {'score': score}
tuner = tune.Tuner(Trainable, run_config=train.RunConfig(stop={'training_iteration': 20}, checkpoint_config=train.CheckpointConfig(checkpoint_at_end=False)), param_space={'a': 2, 'b': 4})
results = tuner.fit()