from rest_framework import status
from sentry.integrations.slack.message_builder.disconnected import DISCONNECTED_MESSAGE
from sentry.testutils.helpers import get_response_text
from sentry.testutils.silo import region_silo_test
from tests.sentry.integrations.slack.webhooks.commands import SlackCommandsTest

@region_silo_test(stable=True)
class SlackCommandsPostTest(SlackCommandsTest):

    def test_invalid_signature(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_error_response(status_code=status.HTTP_401_UNAUTHORIZED)

    def test_missing_team(self):
        if False:
            while True:
                i = 10
        self.get_slack_response({'text': ''}, status_code=status.HTTP_400_BAD_REQUEST)

    def test_idp_does_not_exist(self):
        if False:
            return 10
        'Test that get_identity fails if we cannot find a matching idp.'
        data = self.send_slack_message('', team_id='slack:2')
        assert DISCONNECTED_MESSAGE in get_response_text(data)