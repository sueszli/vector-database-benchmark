class TemporaryAttrValue:

    def __init__(self, obj, attr, value):
        if False:
            while True:
                i = 10
        self.obj = obj
        self.attr = attr
        self.value = value
        self.value_before = None

    def __enter__(self):
        if False:
            print('Hello World!')
        self.value_before = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(self, *_):
        if False:
            print('Hello World!')
        setattr(self.obj, self.attr, self.value_before)
        self.value_before = None

def isbound(method_or_fn):
    if False:
        for i in range(10):
            print('nop')
    try:
        return method_or_fn.__self__ is not None
    except AttributeError:
        try:
            return method_or_fn.__self__ is not None
        except AttributeError:
            return False