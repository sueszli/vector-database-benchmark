class KeyMaker(object):

    def __init__(self, prefix):
        if False:
            return 10
        self.prefix = prefix
        self.num = -1

    def get_new(self):
        if False:
            print('Hello World!')
        self.num += 1
        if self.prefix == '':
            return self.num
        else:
            return '{}{:06d}'.format(self.prefix, self.num)