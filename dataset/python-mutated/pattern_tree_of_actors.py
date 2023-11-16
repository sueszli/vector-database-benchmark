import ray

@ray.remote(num_cpus=1)
class Trainer:

    def __init__(self, hyperparameter, data):
        if False:
            for i in range(10):
                print('nop')
        self.hyperparameter = hyperparameter
        self.data = data

    def fit(self):
        if False:
            i = 10
            return i + 15
        return self.data * self.hyperparameter

@ray.remote(num_cpus=1)
class Supervisor:

    def __init__(self, hyperparameter, data):
        if False:
            return 10
        self.trainers = [Trainer.remote(hyperparameter, d) for d in data]

    def fit(self):
        if False:
            for i in range(10):
                print('nop')
        return ray.get([trainer.fit.remote() for trainer in self.trainers])
data = [1, 2, 3]
supervisor1 = Supervisor.remote(1, data)
supervisor2 = Supervisor.remote(2, data)
model1 = supervisor1.fit.remote()
model2 = supervisor2.fit.remote()
assert ray.get(model1) == [1, 2, 3]
assert ray.get(model2) == [2, 4, 6]