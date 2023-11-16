class NonAsciiKeywordNames:

    def __init__(self, include_latin1=False):
        if False:
            while True:
                i = 10
        self.names = ['Unicode nön-äscïï', '☃', 'UTF-8 nön-äscïï'.encode('UTF-8')]
        if include_latin1:
            self.names.append('Latin1 nön-äscïï'.encode('latin1'))

    def get_keyword_names(self):
        if False:
            i = 10
            return i + 15
        return self.names

    def run_keyword(self, name, args):
        if False:
            i = 10
            return i + 15
        return name