import pytest
from unittest import mock
from apprise.attachment.AttachBase import AttachBase
import logging
logging.disable(logging.CRITICAL)

def test_mimetype_initialization():
    if False:
        return 10
    '\n    API: AttachBase() mimetype initialization\n\n    '
    with mock.patch('mimetypes.init') as mock_init:
        with mock.patch('mimetypes.inited', True):
            AttachBase()
            assert mock_init.call_count == 0
    with mock.patch('mimetypes.init') as mock_init:
        with mock.patch('mimetypes.inited', False):
            AttachBase()
            assert mock_init.call_count == 1

def test_attach_base():
    if False:
        while True:
            i = 10
    '\n    API: AttachBase()\n\n    '
    with pytest.raises(TypeError):
        AttachBase(**{'mimetype': 'invalid'})
    AttachBase(**{'mimetype': 'image/png'})
    obj = AttachBase()
    str(obj)
    with pytest.raises(NotImplementedError):
        obj.download()
    with pytest.raises(NotImplementedError):
        obj.name
    with pytest.raises(NotImplementedError):
        obj.path
    with pytest.raises(NotImplementedError):
        obj.mimetype
    assert AttachBase.parse_url(url='invalid://') is None
    results = AttachBase.parse_url(url='file://relative/path')
    assert isinstance(results, dict)
    assert results.get('mimetype') is None
    results = AttachBase.parse_url(url='file://relative/path?mime=image/jpeg')
    assert isinstance(results, dict)
    assert results.get('mimetype') == 'image/jpeg'
    assert str(results)