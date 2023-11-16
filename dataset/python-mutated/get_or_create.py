import ray

@ray.remote
class Greeter:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

    def say_hello(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value
a = Greeter.options(name='g1', get_if_exists=True).remote('Old Greeting')
assert ray.get(a.say_hello.remote()) == 'Old Greeting'
b = Greeter.options(name='g1', get_if_exists=True).remote('New Greeting')
assert ray.get(b.say_hello.remote()) == 'Old Greeting'