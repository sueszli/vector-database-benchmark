class InvalidKeywordNames:

    def __init__(self, hybrid=False):
        if False:
            for i in range(10):
                print('nop')
        if not hybrid:
            self.run_keyword = lambda *args: None

    def get_keyword_names(self):
        if False:
            while True:
                i = 10
        return 1