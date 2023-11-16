import ray

@ray.remote
class MyActor:

    def __init__(self, index):
        if False:
            while True:
                i = 10
        self.index = index

    def foo(self):
        if False:
            for i in range(10):
                print('nop')
        print('hello there')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'MyActor(index={self.index})'
a = MyActor.remote(1)
b = MyActor.remote(2)
ray.get(a.foo.remote())
ray.get(b.foo.remote())