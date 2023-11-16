class test(list):

    def __init__(self, *args):
        if False:
            return 10
        list.__init__(self, *args)

    @staticmethod
    def method():
        if False:
            while True:
                i = 10
        pass
a = [1, 2, 3]
b = [1, 2, 4]
name = True