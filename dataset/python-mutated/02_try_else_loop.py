def test_constructor():
    if False:
        return 10
    for bad in ('0', 0.0, 0j, (), [], {}, None):
        try:
            raise TypeError(bad)
        except TypeError:
            pass
        else:
            assert False, "%r didn't raise TypeError" % bad
        try:
            raise TypeError(bad)
        except TypeError:
            pass
        else:
            assert False, "%r didn't raise TypeError" % bad
test_constructor()