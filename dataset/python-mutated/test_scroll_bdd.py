import pytest
import pytest_bdd as bdd
bdd.scenarios('scroll.feature')

@pytest.fixture(autouse=True)
def turn_on_scroll_logging(quteproc):
    if False:
        print('Hello World!')
    quteproc.turn_on_scroll_logging(no_scroll_filtering=True)