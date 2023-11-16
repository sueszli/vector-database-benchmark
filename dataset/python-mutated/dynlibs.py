class _BaseDynamicLibrary:

    def get_keyword_names(self):
        if False:
            print('Hello World!')
        return []

    def run_keyword(self, name, *args):
        if False:
            i = 10
            return i + 15
        return None

class StaticDocsLib(_BaseDynamicLibrary):
    """This is lib intro."""

    def __init__(self, some=None, args=[]):
        if False:
            print('Hello World!')
        'Init doc.'

class DynamicDocsLib(_BaseDynamicLibrary):

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_keyword_documentation(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name == '__intro__':
            return 'Dynamic intro doc.'
        if name == '__init__':
            return 'Dynamic init doc.'
        return ''

class StaticAndDynamicDocsLib(_BaseDynamicLibrary):
    """This is static doc."""

    def __init__(self, an_arg=None):
        if False:
            while True:
                i = 10
        'This is static doc.'

    def get_keyword_documentation(self, name):
        if False:
            while True:
                i = 10
        if name == '__intro__':
            return 'dynamic override'
        if name == '__init__':
            return 'dynamic override'
        return ''

class FailingDynamicDocLib(_BaseDynamicLibrary):
    """intro-o-o"""

    def __init__(self):
        if False:
            print('Hello World!')
        'initoo-o-o'

    def get_keyword_documentation(self, name):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('Failing in get_keyword_documentation')