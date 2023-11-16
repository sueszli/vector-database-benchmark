x = 1

def __dir__():
    if False:
        return 10
    return ['a', 'b', 'c']

def __getattr__(name):
    if False:
        i = 10
        return i + 15
    if name == 'yolo':
        raise AttributeError('Deprecated, use whatever instead')
    return f'There is {name}'
y = 2