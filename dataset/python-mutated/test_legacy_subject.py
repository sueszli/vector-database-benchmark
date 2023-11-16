import orjson
from zerver.lib.test_classes import ZulipTestCase

class LegacySubjectTest(ZulipTestCase):

    def test_legacy_subject(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.login('hamlet')
        payload = dict(type='stream', to=orjson.dumps('Verona').decode(), content='Test message')
        payload['subject'] = 'whatever'
        result = self.client_post('/json/messages', payload)
        self.assert_json_success(result)
        payload['topic'] = 'whatever'
        result = self.client_post('/json/messages', payload)
        self.assert_json_error(result, "Can't decide between 'topic' and 'subject' arguments")