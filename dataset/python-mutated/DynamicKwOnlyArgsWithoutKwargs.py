class DynamicKwOnlyArgsWithoutKwargs:

    def get_keyword_names(self):
        if False:
            while True:
                i = 10
        return ['No kwargs']

    def get_keyword_arguments(self, name):
        if False:
            i = 10
            return i + 15
        return ['*', 'kwo']

    def run_keyword(self, name, args):
        if False:
            i = 10
            return i + 15
        raise RuntimeError('Should not be executed!')