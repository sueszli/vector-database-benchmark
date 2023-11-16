import json
import pytest
import pytest_bdd as bdd

@pytest.fixture(autouse=True)
def init_fake_clipboard(quteproc):
    if False:
        print('Hello World!')
    'Make sure the fake clipboard will be used.'
    quteproc.send_cmd(':debug-set-fake-clipboard')

@bdd.then(bdd.parsers.parse('"{text}" should be found'))
def check_found_text(request, quteproc, text):
    if False:
        for i in range(10):
            print('nop')
    if request.config.webengine:
        return
    quteproc.send_cmd(':yank selection')
    quteproc.wait_for(message='Setting fake clipboard: {}'.format(json.dumps(text)))
bdd.scenarios('search.feature')