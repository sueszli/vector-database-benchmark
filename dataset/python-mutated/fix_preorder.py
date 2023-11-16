from lib2to3.fixer_base import BaseFix

class FixPreorder(BaseFix):
    order = 'pre'

    def match(self, node):
        if False:
            while True:
                i = 10
        return False