def test_exc(code, exc):
    if False:
        while True:
            i = 10
    try:
        exec(code)
        print('no exception')
    except exc:
        print('right exception')
    except:
        print('wrong exception')
try:
    1 << 70 in 1
except TypeError:
    print('TypeError')