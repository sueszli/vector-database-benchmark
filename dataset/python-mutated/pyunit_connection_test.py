import sys
sys.path.insert(1, '../../')
import h2o
from tests import pyunit_utils
import requests
from h2o.exceptions import H2OConnectionError

def test_server_is_responding_with_unexpected_data():
    if False:
        print('Hello World!')

    def empty_metadata_request(method, url, **kwargs):
        if False:
            while True:
                i = 10
        if method == 'GET' and url.find('/3/Metadata/schemas') != -1:
            response = requests.Response()
            response.status_code = 200
            return response
    try:
        requests.request_orig = requests.request
        requests.request = empty_metadata_request
        h2o.connect()
        assert False, 'Connection should fail'
    except H2OConnectionError as e:
        print(str(e))
        assert 'Unexpected API output. Please verify server url and port.' in str(e)
    finally:
        requests.request = requests.request_orig
if __name__ == '__main__':
    pyunit_utils.standalone_test(test_server_is_responding_with_unexpected_data)
else:
    test_server_is_responding_with_unexpected_data()