def return_return_bug(foo):
    if False:
        while True:
            i = 10
    if foo == 'say_hello':
        return 'hello'
    return 'world'
assert return_return_bug('say_hello') == 'hello'
assert return_return_bug('world') == 'world'