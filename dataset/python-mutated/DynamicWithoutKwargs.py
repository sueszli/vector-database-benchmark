from helper import pretty
KEYWORDS = {'One Arg': ['arg'], 'Two Args': ['first', 'second'], 'Four Args': ['a=1', ('b', '2'), ('c', 3), ('d', 4)], 'Defaults w/ Specials': ['a=${notvar}', 'b=\n', 'c=\\n', 'd=\\'], 'Args & Varargs': ['a', 'b=default', '*varargs'], 'Nön-ÄSCII names': ['nönäscii', '官话']}

class DynamicWithoutKwargs:

    def __init__(self, **extra):
        if False:
            i = 10
            return i + 15
        self.keywords = dict(KEYWORDS, **extra)

    def get_keyword_names(self):
        if False:
            print('Hello World!')
        return self.keywords.keys()

    def run_keyword(self, kw_name, args):
        if False:
            print('Hello World!')
        return self._pretty(*args)

    def _pretty(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return pretty(*args, **kwargs)

    def get_keyword_arguments(self, kw_name):
        if False:
            return 10
        return self.keywords[kw_name]