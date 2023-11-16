from __future__ import annotations
import os
import textwrap
import conftest
import pytest
os.chdir(os.path.join('..', '..'))

@pytest.fixture(scope='session')
def test_name() -> str:
    if False:
        for i in range(10):
            print('nop')
    return 'ppai/weather-overview'

def test_overview(project: str) -> None:
    if False:
        return 10
    conftest.run_notebook_parallel(os.path.join('notebooks', '1-overview.ipynb'), prelude=textwrap.dedent(f'            # Google Cloud resources.\n            project = {repr(project)}\n            '), sections={'# 🧭 Overview': {}})