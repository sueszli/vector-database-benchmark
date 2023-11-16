class Test(object):

    def run(self):
        if False:
            return 10
        '\n        >>> Test().run()\n        NameError1\n        NameError2\n        found mangled\n        '
        try:
            print(__something)
        except NameError:
            print('NameError1')
        globals()['__something'] = 'found unmangled'
        try:
            print(__something)
        except NameError:
            print('NameError2')
        globals()['_Test__something'] = 'found mangled'
        try:
            print(__something)
        except NameError:
            print('NameError3')