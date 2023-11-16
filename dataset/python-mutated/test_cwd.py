import pytest
from e2b import CurrentWorkingDirectoryDoesntExistException, Sandbox

def test_process_cwd():
    if False:
        i = 10
        return i + 15
    sandbox = Sandbox(cwd='/code/app')
    proc = sandbox.process.start('pwd')
    output = proc.wait()
    assert output.stdout == '/code/app'
    sandbox.close()

def test_filesystem_cwd():
    if False:
        i = 10
        return i + 15
    sandbox = Sandbox(cwd='/code/app')
    sandbox.filesystem.write('hello.txt', 'Hello VM!')
    proc = sandbox.process.start('cat /code/app/hello.txt')
    output = proc.wait()
    assert output.stdout == 'Hello VM!'
    sandbox.close()

def test_change_cwd():
    if False:
        i = 10
        return i + 15
    sandbox = Sandbox(cwd='/code/app')
    sandbox.cwd = '/home/user'
    proc = sandbox.process.start('pwd')
    output = proc.wait()
    assert output.stdout == '/home/user'
    sandbox.filesystem.write('hello.txt', 'Hello VM!')
    proc = sandbox.process.start('cat /home/user/hello.txt')
    output = proc.wait()
    assert output.stdout == 'Hello VM!'
    sandbox.close()

def test_initial_cwd_with_tilde():
    if False:
        while True:
            i = 10
    sandbox = Sandbox(cwd='~/code/')
    proc = sandbox.process.start('pwd')
    output = proc.wait()
    assert output.stdout == '/home/user/code'
    sandbox.close()

def test_relative_paths():
    if False:
        return 10
    sandbox = Sandbox(cwd='/home/user')
    sandbox.filesystem.make_dir('./code')
    sandbox.filesystem.write('./code/hello.txt', 'Hello Vasek!')
    proc = sandbox.process.start('cat /home/user/code/hello.txt')
    output = proc.wait()
    assert output.stdout == 'Hello Vasek!'
    sandbox.filesystem.write('../../hello.txt', 'Hello Tom!')
    proc = sandbox.process.start('cat /hello.txt')
    output = proc.wait()
    assert output.stdout == 'Hello Tom!'
    sandbox.close()

def test_warnings():
    if False:
        print('Hello World!')
    sandbox = Sandbox()
    with pytest.warns(Warning):
        sandbox.filesystem.write('./hello.txt', 'Hello Vasek!')
    with pytest.warns(Warning):
        sandbox.filesystem.write('../hello.txt', 'Hello Vasek!')
    with pytest.warns(Warning):
        sandbox.filesystem.write('~/hello.txt', 'Hello Vasek!')
    sandbox.close()

def test_doesnt_exists():
    if False:
        while True:
            i = 10
    sandbox = Sandbox()
    with pytest.raises(CurrentWorkingDirectoryDoesntExistException):
        sandbox.process.start('ls', cwd='/this/doesnt/exist')
    sandbox.close()