import pytest

@pytest.mark.parametrize('action', ['submit', 'submit_pkg_info'])
def test_removed_upload_apis(webtest, action):
    if False:
        for i in range(10):
            print('nop')
    resp = webtest.post(f'/legacy/?:action={action}', status=410)
    assert resp.status == '410 Project pre-registration is no longer required or supported, upload your files instead.'

def test_remove_doc_upload(webtest):
    if False:
        while True:
            i = 10
    resp = webtest.post('/legacy/?:action=doc_upload', status=410)
    assert resp.status == '410 Uploading documentation is no longer supported, we recommend using https://readthedocs.org/.'

def test_doap(webtest):
    if False:
        return 10
    resp = webtest.get('/pypi?:action=doap&name=foo&version=1.0', status=410)
    assert resp.status == '410 DOAP is no longer supported.'