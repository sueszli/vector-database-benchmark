@decorator
def fun():
    if False:
        i = 10
        return i + 15
    return 42

@decorator(1, 2, 3)
def grumble():
    if False:
        for i in range(10):
            print('nop')
    return 'wahoo'

@decorator('oh')
@decorator('my')
def grr():
    if False:
        while True:
            i = 10
    return 0

@decorator(10)
class ifed:

    def __init__():
        if False:
            for i in range(10):
                print('nop')
        return None

class room(1, 2, 3):

    @look('up')
    def __init__():
        if False:
            i = 10
            return i + 15
        return None