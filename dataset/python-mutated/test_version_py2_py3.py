"""Dependency injector common unit tests."""
from dependency_injector import __version__

def test_version_follows_semantic_versioning():
    if False:
        while True:
            i = 10
    assert len(__version__.split('.')) == 3