import pytest
import pytest_bdd as bdd
bdd.scenarios('yankpaste.feature')

@pytest.fixture(autouse=True)
def init_fake_clipboard(quteproc):
    if False:
        return 10
    'Make sure the fake clipboard will be used.'
    quteproc.send_cmd(':debug-set-fake-clipboard')

@bdd.when(bdd.parsers.parse('I insert "{value}" into the text field'))
def set_text_field(quteproc, value):
    if False:
        for i in range(10):
            print('nop')
    quteproc.send_cmd(":jseval --world=0 set_text('{}')".format(value))
    quteproc.wait_for_js('textarea set to: ' + value)