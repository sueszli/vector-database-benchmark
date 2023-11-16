import orjson
from zerver.lib.exceptions import JsonableError
from zerver.lib.recipient_parsing import extract_direct_message_recipient_ids, extract_stream_id
from zerver.lib.test_classes import ZulipTestCase

class TestRecipientParsing(ZulipTestCase):

    def test_extract_stream_id(self) -> None:
        if False:
            return 10
        stream_id = extract_stream_id('1')
        self.assertEqual(stream_id, 1)
        with self.assertRaisesRegex(JsonableError, 'Invalid data type for stream ID'):
            extract_stream_id('1,2')
        with self.assertRaisesRegex(JsonableError, 'Invalid data type for stream ID'):
            extract_stream_id('[1]')
        with self.assertRaisesRegex(JsonableError, 'Invalid data type for stream ID'):
            extract_stream_id('general')

    def test_extract_recipient_ids(self) -> None:
        if False:
            i = 10
            return i + 15
        user_ids = '[3,2,1]'
        result = sorted(extract_direct_message_recipient_ids(user_ids))
        self.assertEqual(result, [1, 2, 3])
        user_ids = orjson.dumps([3, 3, 12]).decode()
        result = sorted(extract_direct_message_recipient_ids(user_ids))
        self.assertEqual(result, [3, 12])
        user_ids = '1, 12'
        with self.assertRaisesRegex(JsonableError, 'Invalid data type for recipients'):
            extract_direct_message_recipient_ids(user_ids)
        user_ids = orjson.dumps(dict(recipient=12)).decode()
        with self.assertRaisesRegex(JsonableError, 'Invalid data type for recipients'):
            extract_direct_message_recipient_ids(user_ids)
        user_ids = orjson.dumps([3, 4, 'eeshan@example.com']).decode()
        with self.assertRaisesRegex(JsonableError, 'Recipient list may only contain user IDs'):
            extract_direct_message_recipient_ids(user_ids)