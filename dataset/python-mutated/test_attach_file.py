import re
import time
import urllib
from unittest import mock
from os.path import dirname
from os.path import join
from apprise.attachment.AttachBase import AttachBase
from apprise.attachment.AttachFile import AttachFile
from apprise import AppriseAttachment
from apprise.common import ContentLocation
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = join(dirname(__file__), 'var')

def test_attach_file_parse_url():
    if False:
        while True:
            i = 10
    '\n    API: AttachFile().parse_url()\n\n    '
    assert AttachFile.parse_url('garbage://') is None
    assert AttachFile.parse_url('file://') is None

def test_file_expiry(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    '\n    API: AttachFile Expiry\n    '
    path = join(TEST_VAR_DIR, 'apprise-test.gif')
    image = tmpdir.mkdir('apprise_file').join('test.jpg')
    with open(path, 'rb') as data:
        image.write(data)
    aa = AppriseAttachment.instantiate(str(image), cache=30)
    assert aa.exists()
    assert aa.exists()
    with mock.patch('time.time', return_value=time.time() + 31):
        assert aa.exists()
    with mock.patch('time.time', side_effect=OSError):
        assert aa.exists()

def test_attach_file():
    if False:
        while True:
            i = 10
    '\n    API: AttachFile()\n\n    '
    path = join(TEST_VAR_DIR, 'apprise-test.gif')
    response = AppriseAttachment.instantiate(path)
    assert isinstance(response, AttachFile)
    assert response.path == path
    assert response.name == 'apprise-test.gif'
    assert response.mimetype == 'image/gif'
    assert response.download()
    path_in_url = urllib.parse.quote(path)
    assert response.url().startswith('file://{}'.format(path_in_url))
    assert re.search('[?&]mime=', response.url()) is None
    assert re.search('[?&]name=', response.url()) is None
    response = AppriseAttachment.instantiate(path)
    assert isinstance(response, AttachFile)
    response.location = ContentLocation.INACCESSIBLE
    assert response.path is None
    assert response.download() is False
    response = AppriseAttachment.instantiate(path)
    assert isinstance(response, AttachFile)
    with mock.patch('os.path.getsize', return_value=AttachBase.max_file_size):
        assert response.path == path
    response = AppriseAttachment.instantiate(path)
    assert isinstance(response, AttachFile)
    with mock.patch('os.path.getsize', return_value=AttachBase.max_file_size + 1):
        assert response.path is None
    response = AppriseAttachment.instantiate(path)
    assert isinstance(response, AttachFile)
    with mock.patch('os.path.isfile', return_value=False):
        assert response.path is None
    response = AppriseAttachment.instantiate(path)
    assert response.name == 'apprise-test.gif'
    assert response.path == path
    assert response.mimetype == 'image/gif'
    assert re.search('[?&]mime=', response.url()) is None
    assert re.search('[?&]name=', response.url()) is None
    response = AppriseAttachment.instantiate(path)
    assert isinstance(response, AttachFile)
    with mock.patch('os.path.isfile', return_value=False):
        assert response.name is None
    response = AppriseAttachment.instantiate(path)
    assert response.mimetype == 'image/gif'
    assert response.name == 'apprise-test.gif'
    assert response.path == path
    assert re.search('[?&]mime=', response.url()) is None
    assert re.search('[?&]name=', response.url()) is None
    response = AppriseAttachment.instantiate(path)
    assert isinstance(response, AttachFile)
    with mock.patch('os.path.isfile', return_value=False):
        assert response.mimetype is None
        assert response.name is None
        assert response.path is None
    response = AppriseAttachment.instantiate('file://{}?mime={}&name={}'.format(path, 'image/jpeg', 'test.jpeg'))
    assert isinstance(response, AttachFile)
    assert response.path == path
    assert response.name == 'test.jpeg'
    assert response.mimetype == 'image/jpeg'
    assert re.search('[?&]mime=image%2Fjpeg', response.url(), re.I)
    assert re.search('[?&]name=test\\.jpeg', response.url(), re.I)
    aa = AppriseAttachment(location=ContentLocation.HOSTED)
    assert aa.add(path) is False