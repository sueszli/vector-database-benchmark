def testit(a, b):
    if False:
        print('Hello World!')
    if a:
        if not b:
            raise AssertionError('test JUMP_ABSOLUTE to next instruction')

def testit2(a, b):
    if False:
        return 10
    if a:
        if not b:
            raise AssertionError('test with dead code after raise')
            x = 10
testit(False, True)
testit(False, False)
testit(True, True)
testit2(False, True)
testit2(False, False)
testit2(True, True)