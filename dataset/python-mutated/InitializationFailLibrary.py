class InitializationFailLibrary:

    def __init__(self, arg1='default 1', arg2='default 2'):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Initialization failed with arguments %r and %r!' % (arg1, arg2))