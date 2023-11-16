from pipelines.models.singleton import Singleton

class SingletonChild(Singleton):

    def __init__(self):
        if False:
            while True:
                i = 10
        if not self._initialized[self.__class__]:
            self.value = 'initialized'
            self._initialized[self.__class__] = True

def test_singleton_instance():
    if False:
        print('Hello World!')
    instance1 = SingletonChild()
    instance2 = SingletonChild()
    assert instance1 is instance2

def test_singleton_unique_per_subclass():
    if False:
        while True:
            i = 10

    class AnotherSingletonChild(Singleton):
        pass
    instance1 = SingletonChild()
    instance2 = AnotherSingletonChild()
    assert instance1 is not instance2

def test_singleton_initialized():
    if False:
        return 10
    instance = SingletonChild()
    instance.value
    assert instance._initialized[SingletonChild]