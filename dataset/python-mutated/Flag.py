from collections import defaultdict

class Flag(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.valid_flags = set(['admin', 'async_run', 'no_multiuser'])
        self.db = defaultdict(set)

    def __getattr__(self, key):
        if False:
            i = 10
            return i + 15

        def func(f):
            if False:
                return 10
            if key not in self.valid_flags:
                raise Exception('Invalid flag: %s (valid: %s)' % (key, self.valid_flags))
            self.db[f.__name__].add(key)
            return f
        return func
flag = Flag()