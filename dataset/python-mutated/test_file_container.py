import pytest
from PIL import ContainerIO, Image
from .helper import hopper
TEST_FILE = 'Tests/images/dummy.container'

def test_sanity():
    if False:
        while True:
            i = 10
    dir(Image)
    dir(ContainerIO)

def test_isatty():
    if False:
        print('Hello World!')
    with hopper() as im:
        container = ContainerIO.ContainerIO(im, 0, 0)
    assert container.isatty() is False

def test_seek_mode_0():
    if False:
        while True:
            i = 10
    mode = 0
    with open(TEST_FILE, 'rb') as fh:
        container = ContainerIO.ContainerIO(fh, 22, 100)
        container.seek(33, mode)
        container.seek(33, mode)
        assert container.tell() == 33

def test_seek_mode_1():
    if False:
        return 10
    mode = 1
    with open(TEST_FILE, 'rb') as fh:
        container = ContainerIO.ContainerIO(fh, 22, 100)
        container.seek(33, mode)
        container.seek(33, mode)
        assert container.tell() == 66

def test_seek_mode_2():
    if False:
        print('Hello World!')
    mode = 2
    with open(TEST_FILE, 'rb') as fh:
        container = ContainerIO.ContainerIO(fh, 22, 100)
        container.seek(33, mode)
        container.seek(33, mode)
        assert container.tell() == 100

@pytest.mark.parametrize('bytesmode', (True, False))
def test_read_n0(bytesmode):
    if False:
        print('Hello World!')
    with open(TEST_FILE, 'rb' if bytesmode else 'r') as fh:
        container = ContainerIO.ContainerIO(fh, 22, 100)
        container.seek(81)
        data = container.read()
        if bytesmode:
            data = data.decode()
        assert data == '7\nThis is line 8\n'

@pytest.mark.parametrize('bytesmode', (True, False))
def test_read_n(bytesmode):
    if False:
        for i in range(10):
            print('nop')
    with open(TEST_FILE, 'rb' if bytesmode else 'r') as fh:
        container = ContainerIO.ContainerIO(fh, 22, 100)
        container.seek(81)
        data = container.read(3)
        if bytesmode:
            data = data.decode()
        assert data == '7\nT'

@pytest.mark.parametrize('bytesmode', (True, False))
def test_read_eof(bytesmode):
    if False:
        return 10
    with open(TEST_FILE, 'rb' if bytesmode else 'r') as fh:
        container = ContainerIO.ContainerIO(fh, 22, 100)
        container.seek(100)
        data = container.read()
        if bytesmode:
            data = data.decode()
        assert data == ''

@pytest.mark.parametrize('bytesmode', (True, False))
def test_readline(bytesmode):
    if False:
        print('Hello World!')
    with open(TEST_FILE, 'rb' if bytesmode else 'r') as fh:
        container = ContainerIO.ContainerIO(fh, 0, 120)
        data = container.readline()
        if bytesmode:
            data = data.decode()
        assert data == 'This is line 1\n'

@pytest.mark.parametrize('bytesmode', (True, False))
def test_readlines(bytesmode):
    if False:
        return 10
    expected = ['This is line 1\n', 'This is line 2\n', 'This is line 3\n', 'This is line 4\n', 'This is line 5\n', 'This is line 6\n', 'This is line 7\n', 'This is line 8\n']
    with open(TEST_FILE, 'rb' if bytesmode else 'r') as fh:
        container = ContainerIO.ContainerIO(fh, 0, 120)
        data = container.readlines()
        if bytesmode:
            data = [line.decode() for line in data]
        assert data == expected