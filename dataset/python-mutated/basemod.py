class Popen(object):

    def __init__(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        print(repr(args), repr(kw))