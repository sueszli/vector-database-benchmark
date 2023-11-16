import mock
import pytest
import webtest
import blobstore

@pytest.fixture
def app(testbed):
    if False:
        while True:
            i = 10
    return webtest.TestApp(blobstore.app)

def test_img(app):
    if False:
        print('Hello World!')
    with mock.patch('blobstore.images') as mock_images:
        with mock.patch('blobstore.blobstore') as mock_blobstore:
            mock_blobstore.get.return_value = b'123'
            mock_images.resize.return_value = 'asdf'
            mock_images.im_feeling_lucky.return_value = 'gsdf'
            response = app.get('/img?blob_key=123')
            assert response.status_int == 200

def test_img_missing(app):
    if False:
        for i in range(10):
            print('nop')
    app.get('/img?blob_key=123', status=404)

def test_no_img_id(app):
    if False:
        i = 10
        return i + 15
    app.get('/img', status=404)

def test_url_redirect(app):
    if False:
        print('Hello World!')
    with mock.patch('blobstore.images') as mock_images:
        with mock.patch('blobstore.blobstore') as mock_blobstore:
            mock_blobstore.get.return_value = b'123'
            mock_images.get_serving_url.return_value = 'http://lh3.ggpht.com/X'
            response = app.get('/redirect?blob_key=123')
            assert response.status_int == 302

def test_url_redirect_missing(app):
    if False:
        return 10
    app.get('/redirect?blob_key=123', status=404)

def test_url_redirect_no_key(app):
    if False:
        return 10
    app.get('/redirect', status=404)