from __future__ import annotations
from pre_commit_hooks.check_toml import main

def test_toml_bad(tmpdir):
    if False:
        i = 10
        return i + 15
    filename = tmpdir.join('f')
    filename.write('\nkey = # INVALID\n\n= "no key name"  # INVALID\n')
    ret = main((str(filename),))
    assert ret == 1

def test_toml_good(tmpdir):
    if False:
        print('Hello World!')
    filename = tmpdir.join('f')
    filename.write('\n# This is a TOML document.\n\ntitle = "TOML Example"\n\n[owner]\nname = "John"\ndob = 1979-05-27T07:32:00-08:00 # First class dates\n')
    ret = main((str(filename),))
    assert ret == 0

def test_toml_good_unicode(tmpdir):
    if False:
        i = 10
        return i + 15
    filename = tmpdir.join('f')
    filename.write_binary('letter = "☃"\n'.encode())
    ret = main((str(filename),))
    assert ret == 0