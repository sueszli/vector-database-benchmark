"""`Factory` provider - passing injections to the underlying providers example."""
from dependency_injector import containers, providers

class Regularizer:

    def __init__(self, alpha: float) -> None:
        if False:
            while True:
                i = 10
        self.alpha = alpha

class Loss:

    def __init__(self, regularizer: Regularizer) -> None:
        if False:
            i = 10
            return i + 15
        self.regularizer = regularizer

class ClassificationTask:

    def __init__(self, loss: Loss) -> None:
        if False:
            return 10
        self.loss = loss

class Algorithm:

    def __init__(self, task: ClassificationTask) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.task = task

class Container(containers.DeclarativeContainer):
    algorithm_factory = providers.Factory(Algorithm, task=providers.Factory(ClassificationTask, loss=providers.Factory(Loss, regularizer=providers.Factory(Regularizer))))
if __name__ == '__main__':
    container = Container()
    algorithm_1 = container.algorithm_factory(task__loss__regularizer__alpha=0.5)
    assert algorithm_1.task.loss.regularizer.alpha == 0.5
    algorithm_2 = container.algorithm_factory(task__loss__regularizer__alpha=0.7)
    assert algorithm_2.task.loss.regularizer.alpha == 0.7