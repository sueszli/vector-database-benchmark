from typing import Any
from unittest import mock
import pytest
from aiohttp import FormData, web

@pytest.fixture
def buf():
    if False:
        for i in range(10):
            print('nop')
    return bytearray()

@pytest.fixture
def writer(buf: Any):
    if False:
        while True:
            i = 10
    writer = mock.Mock()

    async def write(chunk):
        buf.extend(chunk)
    writer.write.side_effect = write
    return writer

def test_formdata_multipart(buf: Any, writer: Any) -> None:
    if False:
        while True:
            i = 10
    form = FormData()
    assert not form.is_multipart
    form.add_field('test', b'test', filename='test.txt')
    assert form.is_multipart

def test_invalid_formdata_payload() -> None:
    if False:
        return 10
    form = FormData()
    form.add_field('test', object(), filename='test.txt')
    with pytest.raises(TypeError):
        form()

def test_invalid_formdata_params() -> None:
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        FormData('asdasf')

def test_invalid_formdata_params2() -> None:
    if False:
        return 10
    with pytest.raises(TypeError):
        FormData('as')

def test_invalid_formdata_content_type() -> None:
    if False:
        while True:
            i = 10
    form = FormData()
    invalid_vals = [0, 0.1, {}, [], b'foo']
    for invalid_val in invalid_vals:
        with pytest.raises(TypeError):
            form.add_field('foo', 'bar', content_type=invalid_val)

def test_invalid_formdata_filename() -> None:
    if False:
        while True:
            i = 10
    form = FormData()
    invalid_vals = [0, 0.1, {}, [], b'foo']
    for invalid_val in invalid_vals:
        with pytest.raises(TypeError):
            form.add_field('foo', 'bar', filename=invalid_val)

def test_invalid_formdata_content_transfer_encoding() -> None:
    if False:
        while True:
            i = 10
    form = FormData()
    invalid_vals = [0, 0.1, {}, [], b'foo']
    for invalid_val in invalid_vals:
        with pytest.raises(TypeError):
            form.add_field('foo', 'bar', content_transfer_encoding=invalid_val)

async def test_formdata_field_name_is_quoted(buf: Any, writer: Any) -> None:
    form = FormData(charset='ascii')
    form.add_field('email 1', 'xxx@x.co', content_type='multipart/form-data')
    payload = form()
    await payload.write(writer)
    assert b'name="email\\ 1"' in buf

async def test_formdata_field_name_is_not_quoted(buf: Any, writer: Any) -> None:
    form = FormData(quote_fields=False, charset='ascii')
    form.add_field('email 1', 'xxx@x.co', content_type='multipart/form-data')
    payload = form()
    await payload.write(writer)
    assert b'name="email 1"' in buf

async def test_mark_formdata_as_processed(aiohttp_client: Any) -> None:

    async def handler(request):
        return web.Response()
    app = web.Application()
    app.add_routes([web.post('/', handler)])
    client = await aiohttp_client(app)
    data = FormData()
    data.add_field('test', 'test_value', content_type='application/json')
    resp = await client.post('/', data=data)
    assert len(data._writer._parts) == 1
    with pytest.raises(RuntimeError):
        await client.post('/', data=data)
    resp.release()

async def test_formdata_boundary_param() -> None:
    boundary = 'some_boundary'
    form = FormData(boundary=boundary)
    assert form._writer.boundary == boundary