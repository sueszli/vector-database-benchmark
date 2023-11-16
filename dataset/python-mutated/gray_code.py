class GrayCodeGenerator(object):
    """
    Generates and caches gray codes.
    """

    def __init__(self):
        if False:
            return 10
        self.gcs = [0, 1]
        self.lp2 = 2
        self.np2 = 4
        self.i = 2

    def get_gray_code(self, length):
        if False:
            print('Hello World!')
        '\n        Returns a list of gray code of given length.\n        '
        if len(self.gcs) < length:
            self.generate_new_gray_code(length)
        return self.gcs[:length]

    def generate_new_gray_code(self, length):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates new gray code and places into cache.\n        '
        while len(self.gcs) < length:
            if self.i == self.lp2:
                result = self.i + self.i // 2
            else:
                result = self.gcs[2 * self.lp2 - 1 - self.i] + self.lp2
            self.gcs.append(result)
            self.i += 1
            if self.i == self.np2:
                self.lp2 = self.i
                self.np2 = self.i * 2
_gray_code_generator = GrayCodeGenerator()
gray_code = _gray_code_generator.get_gray_code