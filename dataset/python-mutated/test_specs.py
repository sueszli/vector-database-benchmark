import libmambapy

def test_version():
    if False:
        while True:
            i = 10
    ver_str = '1.0'
    ver = libmambapy.Version.parse(ver_str)
    assert str(ver) == ver_str