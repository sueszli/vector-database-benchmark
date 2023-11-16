import pytest
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.dependencies.utils import multipart_incorrect_install_error, multipart_not_installed_error

def test_incorrect_multipart_installed_form(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.delattr('multipart.multipart.parse_options_header', raising=False)
    with pytest.raises(RuntimeError, match=multipart_incorrect_install_error):
        app = FastAPI()

        @app.post('/')
        async def root(username: str=Form()):
            return username

def test_incorrect_multipart_installed_file_upload(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.delattr('multipart.multipart.parse_options_header', raising=False)
    with pytest.raises(RuntimeError, match=multipart_incorrect_install_error):
        app = FastAPI()

        @app.post('/')
        async def root(f: UploadFile=File()):
            return f

def test_incorrect_multipart_installed_file_bytes(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.delattr('multipart.multipart.parse_options_header', raising=False)
    with pytest.raises(RuntimeError, match=multipart_incorrect_install_error):
        app = FastAPI()

        @app.post('/')
        async def root(f: bytes=File()):
            return f

def test_incorrect_multipart_installed_multi_form(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.delattr('multipart.multipart.parse_options_header', raising=False)
    with pytest.raises(RuntimeError, match=multipart_incorrect_install_error):
        app = FastAPI()

        @app.post('/')
        async def root(username: str=Form(), password: str=Form()):
            return username

def test_incorrect_multipart_installed_form_file(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.delattr('multipart.multipart.parse_options_header', raising=False)
    with pytest.raises(RuntimeError, match=multipart_incorrect_install_error):
        app = FastAPI()

        @app.post('/')
        async def root(username: str=Form(), f: UploadFile=File()):
            return username

def test_no_multipart_installed(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.delattr('multipart.__version__', raising=False)
    with pytest.raises(RuntimeError, match=multipart_not_installed_error):
        app = FastAPI()

        @app.post('/')
        async def root(username: str=Form()):
            return username

def test_no_multipart_installed_file(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.delattr('multipart.__version__', raising=False)
    with pytest.raises(RuntimeError, match=multipart_not_installed_error):
        app = FastAPI()

        @app.post('/')
        async def root(f: UploadFile=File()):
            return f

def test_no_multipart_installed_file_bytes(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.delattr('multipart.__version__', raising=False)
    with pytest.raises(RuntimeError, match=multipart_not_installed_error):
        app = FastAPI()

        @app.post('/')
        async def root(f: bytes=File()):
            return f

def test_no_multipart_installed_multi_form(monkeypatch):
    if False:
        return 10
    monkeypatch.delattr('multipart.__version__', raising=False)
    with pytest.raises(RuntimeError, match=multipart_not_installed_error):
        app = FastAPI()

        @app.post('/')
        async def root(username: str=Form(), password: str=Form()):
            return username

def test_no_multipart_installed_form_file(monkeypatch):
    if False:
        return 10
    monkeypatch.delattr('multipart.__version__', raising=False)
    with pytest.raises(RuntimeError, match=multipart_not_installed_error):
        app = FastAPI()

        @app.post('/')
        async def root(username: str=Form(), f: UploadFile=File()):
            return username