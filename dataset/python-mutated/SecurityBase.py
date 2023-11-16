import re

class StockBase:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def valid_code(self, code):
        if False:
            return 10
        pattern = re.search('(\\d{6})', code)
        if pattern:
            code = pattern.group(1)
        else:
            code = None
        return code