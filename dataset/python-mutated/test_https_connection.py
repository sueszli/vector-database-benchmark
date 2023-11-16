import pytest
import ssl
import sys
import trustme
import ray
from ray.job_submission import JobSubmissionClient

@pytest.fixture(scope='session')
def ca():
    if False:
        for i in range(10):
            print('nop')
    return trustme.CA()

@pytest.fixture(scope='session')
def httpserver_ssl_context(ca):
    if False:
        return 10
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    localhost_cert = ca.issue_cert('localhost')
    localhost_cert.configure_cert(context)
    return context

@pytest.fixture(scope='session')
def httpclient_ssl_context(ca):
    if False:
        return 10
    with ca.cert_pem.tempfile() as ca_temp_path:
        return ssl.create_default_context(cafile=ca_temp_path)

def test_mock_https_connection(httpserver, ca):
    if False:
        i = 10
        return i + 15
    'Test connections to a mock HTTPS job submission server.'
    httpserver.expect_request('/api/version').respond_with_json({'ray_version': ray.__version__})
    mock_url = httpserver.url_for('/')
    with pytest.raises(ConnectionError):
        JobSubmissionClient(mock_url)
    JobSubmissionClient(mock_url, verify=False)
    with ca.cert_pem.tempfile() as ca_temp_path:
        JobSubmissionClient(mock_url, verify=ca_temp_path)
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))