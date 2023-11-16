def test1():
    if False:
        i = 10
        return i + 15
    hardcoded = 'hello'
    with something() as foobar:
        print(hardcoded)