ZEN = "\nThe Zen of Python\nBeautiful is better than ugly.\nExplicit is better than implicit.\nSimple is better than complex.\nComplex is better than complicated.\nFlat is better than nested.\nSparse is better than dense.\nReadability counts.\nSpecial cases aren't special enough to break the rules.\nAlthough practicality beats purity.\nErrors should never pass silently.\nUnless explicitly silenced.\nIn the face of ambiguity, refuse the temptation to guess.\nThere should be one-- and preferably only one --obvious way to do it.\nAlthough that way may not be obvious at first unless you're Dutch.\nNow is better than never.\nAlthough never is often better than *right* now.\nIf the implementation is hard to explain, it's a bad idea.\nIf the implementation is easy to explain, it may be a good idea.\nNamespaces are one honking great idea -- let's do more of those!\n"

def test(niter):
    if False:
        for i in range(10):
            print('nop')
    counts = {}
    for _ in range(niter):
        x = ZEN.replace('\n', ' ').split(' ')
        y = ' '.join(x)
        for i in range(50):
            a = ZEN[i:i * 2]
            b = a + 'hello world'
        for c in ZEN:
            i = ord(c)
            c = chr(i)
    return (x[0], a)
bm_params = {(32, 10): (2,), (50, 10): (3,), (100, 10): (6,), (500, 10): (30,), (1000, 10): (60,), (5000, 10): (300,)}

def bm_setup(params):
    if False:
        while True:
            i = 10
    (niter,) = params
    state = None

    def run():
        if False:
            i = 10
            return i + 15
        nonlocal state
        state = test(niter)

    def result():
        if False:
            return 10
        return (niter, state)
    return (run, result)