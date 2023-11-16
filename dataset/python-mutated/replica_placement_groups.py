from ray import serve

@serve.deployment
class D:

    def __call__(self, *args):
        if False:
            i = 10
            return i + 15
        return 'hi'
app = D.bind()