import sys
from remoteserver import RemoteServer, keyword

class BulkLoadRemoteServer(RemoteServer):

    def _register_functions(self):
        if False:
            i = 10
            return i + 15
        '\n        Individual get_keyword_* methods are not registered.\n        This removes the fall back scenario should get_library_information fail.\n        '
        self.register_function(self.get_library_information)
        self.register_function(self.run_keyword)

    def get_library_information(self):
        if False:
            print('Hello World!')
        info_dict = {'__init__': {'doc': '__init__ documentation.'}, '__intro__': {'doc': '__intro__ documentation.'}}
        for kw in self.get_keyword_names():
            info_dict[kw] = dict(args=['arg', '*extra'] if kw == 'some_keyword' else ['arg=None'], doc="Documentation for '%s'." % kw, tags=['tag'], types=['bool'] if kw == 'some_keyword' else ['int'])
        return info_dict

class The10001KeywordsLibrary:

    def __init__(self):
        if False:
            while True:
                i = 10
        for i in range(10000):
            setattr(self, 'keyword_%d' % i, lambda result=str(i): result)

    def some_keyword(self, arg, *extra):
        if False:
            print('Hello World!')
        return 'yes' if arg else 'no'
if __name__ == '__main__':
    BulkLoadRemoteServer(The10001KeywordsLibrary(), *sys.argv[1:])