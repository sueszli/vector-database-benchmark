""" Define a Pytest plugin for a log file fixture.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import IO, Iterator
import pytest
__all__ = ('log_file',)

@pytest.fixture(scope='session')
def log_file(request: pytest.FixtureRequest) -> Iterator[IO[str]]:
    if False:
        for i in range(10):
            print('nop')
    with open(request.config.option.log_file, 'w') as f:
        f.write('')
    with open(request.config.option.log_file, 'a') as f:
        yield f