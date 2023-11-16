import re
_ident_re = re.compile('^[a-zA-Z_-][.a-zA-Z0-9_-]*$')

def ident(x):
    if False:
        for i in range(10):
            print('nop')
    if _ident_re.match(x):
        return x
    raise TypeError

class Matcher:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._patterns = {}
        self._dirty = True

    def __setitem__(self, path, value):
        if False:
            for i in range(10):
                print('nop')
        assert path not in self._patterns, f'duplicate path {path}'
        self._patterns[path] = value
        self._dirty = True

    def __repr__(self):
        if False:
            return 10
        return f'<Matcher {repr(self._patterns)}>'
    path_elt_re = re.compile('^(.?):([a-z0-9_.]+)$')
    type_fns = {'n': int, 'i': ident}

    def __getitem__(self, path):
        if False:
            for i in range(10):
                print('nop')
        if self._dirty:
            self._compile()
        patterns = self._by_length.get(len(path), {})
        for pattern in patterns:
            kwargs = {}
            for (pattern_elt, path_elt) in zip(pattern, path):
                mo = self.path_elt_re.match(pattern_elt)
                if mo:
                    (type_flag, arg_name) = mo.groups()
                    if type_flag:
                        try:
                            type_fn = self.type_fns[type_flag]
                        except Exception:
                            assert type_flag in self.type_fns, f'no such type flag {type_flag}'
                        try:
                            path_elt = type_fn(path_elt)
                        except Exception:
                            break
                    kwargs[arg_name] = path_elt
                elif pattern_elt != path_elt:
                    break
            else:
                return (patterns[pattern], kwargs)
        raise KeyError(f'No match for {repr(path)}')

    def iterPatterns(self):
        if False:
            print('Hello World!')
        return list(self._patterns.items())

    def _compile(self):
        if False:
            print('Hello World!')
        self._by_length = {}
        for (k, v) in self.iterPatterns():
            length = len(k)
            self._by_length.setdefault(length, {})[k] = v