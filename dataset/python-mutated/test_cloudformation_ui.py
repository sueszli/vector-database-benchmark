import requests
from localstack import config
from localstack.testing.pytest import markers
CLOUDFORMATION_UI_PATH = '/_localstack/cloudformation/deploy'

class TestCloudFormationUi:

    @markers.aws.only_localstack
    def test_get_cloudformation_ui(self):
        if False:
            print('Hello World!')
        cfn_ui_url = config.external_service_url() + CLOUDFORMATION_UI_PATH
        response = requests.get(cfn_ui_url)
        assert response.ok
        assert 'content-type' in response.headers
        assert b'LocalStack' in response.content