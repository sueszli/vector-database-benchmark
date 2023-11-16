class RunKeywordLibrary:
    ROBOT_LIBRARY_SCOPE = 'TESTCASE'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.kw_names = ['Run Keyword That Passes', 'Run Keyword That Fails']

    def get_keyword_names(self):
        if False:
            i = 10
            return i + 15
        return self.kw_names

    def run_keyword(self, name, args):
        if False:
            print('Hello World!')
        try:
            method = dict(zip(self.kw_names, [self._passes, self._fails]))[name]
        except KeyError:
            raise AttributeError
        return method(args)

    def _passes(self, args):
        if False:
            for i in range(10):
                print('nop')
        for arg in args:
            print(arg, end=' ')
        return ', '.join(args)

    def _fails(self, args):
        if False:
            return 10
        if not args:
            raise AssertionError('Failure')
        raise AssertionError('Failure: %s' % ' '.join(args))

class GlobalRunKeywordLibrary(RunKeywordLibrary):
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'

class RunKeywordButNoGetKeywordNamesLibrary:

    def run_keyword(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return ' '.join(args)

    def some_other_keyword(self, *args):
        if False:
            print('Hello World!')
        return ' '.join(args)