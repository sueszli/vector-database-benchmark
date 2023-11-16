from ray import serve

@serve.deployment
class D:

    def __call__(self, *args):
        if False:
            while True:
                i = 10
        return 'hi'
app = D.bind()