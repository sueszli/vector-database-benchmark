"""Tests for acme.jose shim."""
import importlib
import sys
import unittest
import pytest

def _test_it(submodule, attribute):
    if False:
        while True:
            i = 10
    if submodule:
        acme_jose_path = 'acme.jose.' + submodule
        josepy_path = 'josepy.' + submodule
    else:
        acme_jose_path = 'acme.jose'
        josepy_path = 'josepy'
    acme_jose_mod = importlib.import_module(acme_jose_path)
    josepy_mod = importlib.import_module(josepy_path)
    assert acme_jose_mod is josepy_mod
    assert getattr(acme_jose_mod, attribute) is getattr(josepy_mod, attribute)
    import josepy
    import acme
    acme_jose_mod = eval(acme_jose_path)
    josepy_mod = eval(josepy_path)
    assert acme_jose_mod is josepy_mod
    assert getattr(acme_jose_mod, attribute) is getattr(josepy_mod, attribute)

def test_top_level():
    if False:
        for i in range(10):
            print('nop')
    _test_it('', 'RS512')

def test_submodules():
    if False:
        for i in range(10):
            print('nop')
    mods_and_attrs = [('b64', 'b64decode'), ('errors', 'Error'), ('interfaces', 'JSONDeSerializable'), ('json_util', 'Field'), ('jwa', 'HS256'), ('jwk', 'JWK'), ('jws', 'JWS'), ('util', 'ImmutableMap')]
    for (mod, attr) in mods_and_attrs:
        _test_it(mod, attr)
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))