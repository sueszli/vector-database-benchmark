def __getattr__():
    if False:
        for i in range(10):
            print('nop')
    'Bad one'
x = 1

def __dir__(bad_sig):
    if False:
        return 10
    return []