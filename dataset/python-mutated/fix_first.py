from lib2to3.fixer_base import BaseFix

class FixFirst(BaseFix):
    run_order = 1

    def match(self, node):
        if False:
            for i in range(10):
                print('nop')
        return False