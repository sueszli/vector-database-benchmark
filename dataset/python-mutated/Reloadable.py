from robot.utils import NormalizedDict
from robot.libraries.BuiltIn import BuiltIn
BUILTIN = BuiltIn()
KEYWORDS = NormalizedDict({'add_keyword': ('name', '*args'), 'remove_keyword': ('name',), 'reload_self': (), 'original 1': ('arg',), 'original 2': ('arg',), 'original 3': ('arg',)})

class Reloadable:

    def get_keyword_names(self):
        if False:
            i = 10
            return i + 15
        return list(KEYWORDS)

    def get_keyword_arguments(self, name):
        if False:
            for i in range(10):
                print('nop')
        return KEYWORDS[name]

    def get_keyword_documentation(self, name):
        if False:
            while True:
                i = 10
        return 'Doc for %s with args %s' % (name, ', '.join(KEYWORDS[name]))

    def run_keyword(self, name, args):
        if False:
            while True:
                i = 10
        print("Running keyword '%s' with arguments %s." % (name, args))
        assert name in KEYWORDS
        if name == 'add_keyword':
            KEYWORDS[args[0]] = args[1:]
        elif name == 'remove_keyword':
            KEYWORDS.pop(args[0])
        elif name == 'reload_self':
            BUILTIN.reload_library(self)
        return name