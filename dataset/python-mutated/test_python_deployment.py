from ray import serve

def echo_server(v):
    if False:
        print('Hello World!')
    return v

@serve.deployment
class Counter(object):

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = int(value)

    def increase(self, delta):
        if False:
            print('Hello World!')
        self.value += int(delta)
        return str(self.value)

    def reconfigure(self, value_str):
        if False:
            print('Hello World!')
        self.value = int(value_str)