import pytest
import pytest_bdd as bdd
bdd.scenarios('utilcmds.feature')

@pytest.fixture(autouse=True)
def turn_on_scroll_logging(quteproc):
    if False:
        print('Hello World!')
    quteproc.turn_on_scroll_logging()