class no_arg_init:
    """No inits here!"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'This doc not shown because there are no arguments.'

    def keyword(self, arg1, arg2):
        if False:
            i = 10
            return i + 15
        'The only lonely keyword.'