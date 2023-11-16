def __badAllowed__():
    if False:
        while True:
            i = 10
    pass

def __stillBad__():
    if False:
        i = 10
        return i + 15
    pass

def nested():
    if False:
        for i in range(10):
            print('nop')

    def __badAllowed__():
        if False:
            return 10
        pass

    def __stillBad__():
        if False:
            for i in range(10):
                print('nop')
        pass