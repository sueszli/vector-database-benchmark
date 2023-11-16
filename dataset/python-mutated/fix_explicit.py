from lib2to3.fixer_base import BaseFix

class FixExplicit(BaseFix):
    explicit = True

    def match(self):
        if False:
            print('Hello World!')
        return False