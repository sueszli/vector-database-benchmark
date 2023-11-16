from __future__ import annotations
from app import return_a_value

def test_dynaconf_is_in_testing_env():
    if False:
        return 10
    assert return_a_value() == 'On Testing'