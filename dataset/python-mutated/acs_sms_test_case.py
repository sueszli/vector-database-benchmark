import os
from devtools_testutils import AzureRecordedTestCase, is_live
from azure.communication.sms._shared.utils import parse_connection_str

class ACSSMSTestCase(AzureRecordedTestCase):

    def setUp(self):
        if False:
            return 10
        self.connection_str = self._get_connection_str()
        if is_live():
            self.phone_number = os.getenv('SMS_PHONE_NUMBER')
            (self.endpoint, _) = parse_connection_str(self.connection_str)
        else:
            (self.endpoint, _) = parse_connection_str(self.connection_str)
            self.phone_number = '+14255550123'
        self._resource_name = self.endpoint.split('.')[0]

    def _get_connection_str(self):
        if False:
            i = 10
            return i + 15
        if self.is_playback():
            return 'endpoint=https://sanitized.communication.azure.com/;accesskey=fake==='
        return os.getenv('COMMUNICATION_LIVETEST_STATIC_CONNECTION_STRING')

    def verify_successful_sms_response(self, sms_response):
        if False:
            for i in range(10):
                print('nop')
        if self.is_live:
            assert sms_response.to == self.phone_number
        assert sms_response.message_id is not None
        assert sms_response.http_status_code == 202
        assert sms_response.error_message is None
        assert sms_response.successful