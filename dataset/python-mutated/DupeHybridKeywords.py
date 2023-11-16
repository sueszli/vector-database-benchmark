class DupeHybridKeywords:
    names = ['defined twice', 'DEFINED TWICE', 'Embedded ${twice}', 'EMBEDDED ${ARG}', 'Exact dupe is ok', 'Exact dupe is ok']

    def get_keyword_names(self):
        if False:
            for i in range(10):
                print('nop')
        return self.names

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name not in self.names:
            raise AttributeError
        return lambda *args: None