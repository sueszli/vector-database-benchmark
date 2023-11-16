from tornado import gen

@gen.coroutine
def hello():
    if False:
        return 10
    yield gen.sleep(0.001)
    raise gen.Return('hello')