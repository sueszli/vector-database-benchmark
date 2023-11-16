import os.path
import logging
import pytest
import pytest_bdd as bdd
bdd.scenarios('sessions.feature')

@pytest.fixture(autouse=True)
def turn_on_scroll_logging(quteproc):
    if False:
        i = 10
        return i + 15
    quteproc.turn_on_scroll_logging()

@bdd.when(bdd.parsers.parse('I have a "{name}" session file:\n{contents}'))
def create_session_file(quteproc, name, contents):
    if False:
        i = 10
        return i + 15
    filename = os.path.join(quteproc.basedir, 'data', 'sessions', name + '.yml')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(contents)

@bdd.when(bdd.parsers.parse('I replace "{pattern}" by "{replacement}" in the "{name}" session file'))
def session_replace(quteproc, server, pattern, replacement, name):
    if False:
        while True:
            i = 10
    quteproc.wait_for(category='message', loglevel=logging.INFO, message='Saved session {}.'.format(name))
    filename = os.path.join(quteproc.basedir, 'data', 'sessions', name + '.yml')
    replacement = replacement.replace('(port)', str(server.port))
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(data.replace(pattern, replacement))

@bdd.then(bdd.parsers.parse('the session {name} should exist'))
def session_should_exist(quteproc, name):
    if False:
        return 10
    filename = os.path.join(quteproc.basedir, 'data', 'sessions', name + '.yml')
    assert os.path.exists(filename)

@bdd.then(bdd.parsers.parse('the session {name} should not exist'))
def session_should_not_exist(quteproc, name):
    if False:
        while True:
            i = 10
    filename = os.path.join(quteproc.basedir, 'data', 'sessions', name + '.yml')
    assert not os.path.exists(filename)