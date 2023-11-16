def test_how_path_joining_works():
    if False:
        print('Hello World!')
    from os.path import join
    assert '/another-absolute' == join('/absolute', '/another-absolute')
    assert '/absolute/relative' == join('/absolute', 'relative')
    assert '/absolute' == join('relative', '/absolute')
    assert 'relative/relative' == join('relative', 'relative')
    assert '/absolute' == join('', '/absolute')