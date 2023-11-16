from ray import serve

@serve.deployment
class A:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        1 / 0
node = A.bind()