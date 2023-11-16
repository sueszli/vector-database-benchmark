class DupeDynamicKeywords:
    names = ['defined twice', 'DEFINED TWICE', 'Embedded ${twice}', 'EMBEDDED ${ARG}', 'Exact dupe is ok', 'Exact dupe is ok']

    def get_keyword_names(self):
        if False:
            print('Hello World!')
        return self.names

    def run_keyword(self, name, args):
        if False:
            return 10
        pass