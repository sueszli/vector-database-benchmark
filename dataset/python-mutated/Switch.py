class Switch(object):

    def __init__(self, value):
        if False:
            return 10
        self.value = value
        self.fall = False

    def __iter__(self):
        if False:
            return 10
        'Return the match method once, then stop'
        yield self.match

    def match(self, *args):
        if False:
            print('Hello World!')
        'Indicate whether or not to enter a case suite'
        result = False
        if self.fall or not args:
            result = True
        elif self.value in args:
            self.fall = True
            result = True
        return result