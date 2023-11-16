from PyInstaller.depend.bindepend import _library_matcher

def test_library_matcher():
    if False:
        return 10
    '\n    Test that _library_matcher() is tolerant to version numbers both before and after the .so suffix but does not\n    allow runaway glob patterns to match anything else.\n    '
    m = _library_matcher('libc')
    assert m('libc.so')
    assert m('libc.dylib')
    assert m('libc.so.1')
    assert not m('libcrypt.so')
    m = _library_matcher('libpng')
    assert m('libpng16.so.16')