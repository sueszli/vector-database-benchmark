class Dict(dict):

    def __init__(self, **kw):
        if False:
            return 10
        super().__init__(**kw)

    def __getattr__(self, key):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'Dict' object has no attribute '%s'" % key)

    def __setattr__(self, key, value):
        if False:
            return 10
        self[key] = value