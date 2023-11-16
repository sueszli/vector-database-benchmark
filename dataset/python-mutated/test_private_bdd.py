import json
import pytest_bdd as bdd
bdd.scenarios('private.feature')

@bdd.then(bdd.parsers.parse('the cookie {name} should be set to {value}'))
def check_cookie(quteproc, name, value):
    if False:
        print('Hello World!')
    "Check if a given cookie is set correctly.\n\n    This assumes we're on the server cookies page.\n    "
    content = quteproc.get_content()
    data = json.loads(content)
    print(data)
    assert data['cookies'][name] == value

@bdd.then(bdd.parsers.parse('the cookie {name} should not be set'))
def check_cookie_not_set(quteproc, name):
    if False:
        return 10
    'Check if a given cookie is not set.'
    content = quteproc.get_content()
    data = json.loads(content)
    print(data)
    assert name not in data['cookies']

@bdd.then(bdd.parsers.parse('the file {name} should not contain "{text}"'))
def check_not_contain(tmpdir, name, text):
    if False:
        while True:
            i = 10
    path = tmpdir / name
    assert text not in path.read()