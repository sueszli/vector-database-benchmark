from __future__ import annotations
import sys
import pytest
from rerun_demo import __main__ as main
pytestmark = pytest.mark.filterwarnings('error')

def test_run_cube() -> None:
    if False:
        i = 10
        return i + 15
    sys.argv = ['prog', '--cube', '--headless']
    main.main()