import time
from copy import deepcopy
from typing import Any, Dict, List, Optional
import orjson
from zerver.lib.test_classes import ZulipTestCase
from zerver.models import Draft

class DraftCreationTests(ZulipTestCase):

    def create_and_check_drafts_for_success(self, draft_dicts: List[Dict[str, Any]], expected_draft_dicts: Optional[List[Dict[str, Any]]]=None) -> None:
        if False:
            return 10
        hamlet = self.example_user('hamlet')
        payload = {'drafts': orjson.dumps(draft_dicts).decode()}
        resp = self.api_post(hamlet, '/api/v1/drafts', payload)
        self.assert_json_success(resp)
        new_draft_dicts = []
        for draft in Draft.objects.filter(user_profile=hamlet).order_by('last_edit_time'):
            draft_dict = draft.to_dict()
            draft_dict.pop('id')
            new_draft_dicts.append(draft_dict)
        if expected_draft_dicts is None:
            expected_draft_dicts = draft_dicts
        self.assertEqual(new_draft_dicts, expected_draft_dicts)

    def create_and_check_drafts_for_error(self, draft_dicts: List[Dict[str, Any]], expected_message: str) -> None:
        if False:
            i = 10
            return i + 15
        hamlet = self.example_user('hamlet')
        initial_count = Draft.objects.count()
        payload = {'drafts': orjson.dumps(draft_dicts).decode()}
        resp = self.api_post(hamlet, '/api/v1/drafts', payload)
        self.assert_json_error(resp, expected_message)
        self.assertEqual(Draft.objects.count(), initial_count)

    def test_require_enable_drafts_synchronization(self) -> None:
        if False:
            i = 10
            return i + 15
        hamlet = self.example_user('hamlet')
        hamlet.enable_drafts_synchronization = False
        hamlet.save()
        payload = {'drafts': '[]'}
        resp = self.api_post(hamlet, '/api/v1/drafts', payload)
        self.assert_json_error(resp, 'User has disabled synchronizing drafts.')

    def test_create_one_stream_draft_properly(self) -> None:
        if False:
            i = 10
            return i + 15
        hamlet = self.example_user('hamlet')
        visible_stream_name = self.get_streams(hamlet)[0]
        visible_stream_id = self.get_stream_id(visible_stream_name)
        draft_dicts = [{'type': 'stream', 'to': [visible_stream_id], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 1595479019}]
        self.create_and_check_drafts_for_success(draft_dicts)

    def test_create_one_personal_message_draft_properly(self) -> None:
        if False:
            return 10
        zoe = self.example_user('ZOE')
        draft_dicts = [{'type': 'private', 'to': [zoe.id], 'topic': 'This topic should be ignored.', 'content': 'What if we made it possible to sync drafts in Zulip?', 'timestamp': 1595479019}]
        expected_draft_dicts = [{'type': 'private', 'to': [zoe.id], 'topic': '', 'content': 'What if we made it possible to sync drafts in Zulip?', 'timestamp': 1595479019}]
        self.create_and_check_drafts_for_success(draft_dicts, expected_draft_dicts)

    def test_create_one_group_personal_message_draft_properly(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        zoe = self.example_user('ZOE')
        othello = self.example_user('othello')
        draft_dicts = [{'type': 'private', 'to': [zoe.id, othello.id], 'topic': 'This topic should be ignored.', 'content': 'What if we made it possible to sync drafts in Zulip?', 'timestamp': 1595479019}]
        expected_draft_dicts = [{'type': 'private', 'to': [zoe.id, othello.id], 'topic': '', 'content': 'What if we made it possible to sync drafts in Zulip?', 'timestamp': 1595479019}]
        self.create_and_check_drafts_for_success(draft_dicts, expected_draft_dicts)

    def test_create_batch_of_drafts_properly(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        hamlet = self.example_user('hamlet')
        visible_stream_name = self.get_streams(hamlet)[0]
        visible_stream_id = self.get_stream_id(visible_stream_name)
        zoe = self.example_user('ZOE')
        othello = self.example_user('othello')
        draft_dicts = [{'type': 'stream', 'to': [visible_stream_id], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 1595479019}, {'type': 'private', 'to': [zoe.id], 'topic': '', 'content': 'What if we made it possible to sync drafts in Zulip?', 'timestamp': 1595479020}, {'type': 'private', 'to': [zoe.id, othello.id], 'topic': '', 'content': 'What if we made it possible to sync drafts in Zulip?', 'timestamp': 1595479021}]
        self.create_and_check_drafts_for_success(draft_dicts)

    def test_missing_timestamps(self) -> None:
        if False:
            while True:
                i = 10
        'If a timestamp is not provided for a draft dict then it should be automatically\n        filled in.'
        hamlet = self.example_user('hamlet')
        visible_stream_name = self.get_streams(hamlet)[0]
        visible_stream_id = self.get_stream_id(visible_stream_name)
        draft_dicts = [{'type': 'stream', 'to': [visible_stream_id], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts."}]
        initial_count = Draft.objects.count()
        current_time = int(time.time())
        payload = {'drafts': orjson.dumps(draft_dicts).decode()}
        resp = self.api_post(hamlet, '/api/v1/drafts', payload)
        ids = orjson.loads(resp.content)['ids']
        self.assert_json_success(resp)
        new_drafts = Draft.objects.filter(id__gte=ids[0])
        self.assertEqual(Draft.objects.count() - initial_count, 1)
        new_draft = new_drafts[0].to_dict()
        self.assertTrue(isinstance(new_draft['timestamp'], int))
        self.assertTrue(new_draft['timestamp'] >= current_time)

    def test_invalid_timestamp(self) -> None:
        if False:
            return 10
        draft_dicts = [{'type': 'stream', 'to': [], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts.", 'timestamp': -10.1}]
        self.create_and_check_drafts_for_error(draft_dicts, 'Timestamp must not be negative.')

    def test_create_non_stream_draft_with_no_recipient(self) -> None:
        if False:
            return 10
        'When "to" is an empty list, the type should become "" as well.'
        draft_dicts = [{'type': 'private', 'to': [], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 1595479019}, {'type': '', 'to': [], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 1595479019}]
        expected_draft_dicts = [{'type': '', 'to': [], 'topic': '', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 1595479019}, {'type': '', 'to': [], 'topic': '', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 1595479019}]
        self.create_and_check_drafts_for_success(draft_dicts, expected_draft_dicts)

    def test_create_stream_draft_with_no_recipient(self) -> None:
        if False:
            print('Hello World!')
        draft_dicts = [{'type': 'stream', 'to': [], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 15954790199}]
        self.create_and_check_drafts_for_error(draft_dicts, 'Must specify exactly 1 stream ID for stream messages')

    def test_create_stream_draft_for_inaccessible_stream(self) -> None:
        if False:
            while True:
                i = 10
        stream = self.make_stream('Secret Society', invite_only=True)
        draft_dicts = [{'type': 'stream', 'to': [stream.id], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 1595479019}]
        self.create_and_check_drafts_for_error(draft_dicts, 'Invalid stream ID')
        draft_dicts = [{'type': 'stream', 'to': [99999999999999], 'topic': 'sync drafts', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 1595479019}]
        self.create_and_check_drafts_for_error(draft_dicts, 'Invalid stream ID')

    def test_create_personal_message_draft_for_non_existing_user(self) -> None:
        if False:
            i = 10
            return i + 15
        draft_dicts = [{'type': 'private', 'to': [99999999999999], 'topic': 'This topic should be ignored.', 'content': 'What if we made it possible to sync drafts in Zulip?', 'timestamp': 1595479019}]
        self.create_and_check_drafts_for_error(draft_dicts, 'Invalid user ID 99999999999999')

    def test_create_draft_with_null_bytes(self) -> None:
        if False:
            return 10
        draft_dicts = [{'type': '', 'to': [], 'topic': 'sync drafts.', 'content': 'Some regular \x00 content here', 'timestamp': 15954790199}]
        self.create_and_check_drafts_for_error(draft_dicts, 'Message must not contain null bytes')
        draft_dicts = [{'type': 'stream', 'to': [10], 'topic': 'thinking about \x00', 'content': "Let's add backend support for syncing drafts.", 'timestamp': 15954790199}]
        self.create_and_check_drafts_for_error(draft_dicts, 'Topic must not contain null bytes')

class DraftEditTests(ZulipTestCase):

    def test_require_enable_drafts_synchronization(self) -> None:
        if False:
            print('Hello World!')
        hamlet = self.example_user('hamlet')
        hamlet.enable_drafts_synchronization = False
        hamlet.save()
        resp = self.api_patch(hamlet, '/api/v1/drafts/1', {'draft': {}})
        self.assert_json_error(resp, 'User has disabled synchronizing drafts.')

    def test_edit_draft_successfully(self) -> None:
        if False:
            i = 10
            return i + 15
        hamlet = self.example_user('hamlet')
        visible_streams = self.get_streams(hamlet)
        stream_a = self.get_stream_id(visible_streams[0])
        stream_b = self.get_stream_id(visible_streams[1])
        draft_dict = {'type': 'stream', 'to': [stream_a], 'topic': 'drafts', 'content': 'The API should be good', 'timestamp': 1595505700}
        resp = self.api_post(hamlet, '/api/v1/drafts', {'drafts': orjson.dumps([draft_dict]).decode()})
        self.assert_json_success(resp)
        new_draft_id = orjson.loads(resp.content)['ids'][0]
        draft_dict['content'] = 'The API needs to be structured yet simple to use.'
        draft_dict['to'] = [stream_b]
        draft_dict['topic'] = 'designing drafts'
        draft_dict['timestamp'] = 1595505800
        resp = self.api_patch(hamlet, f'/api/v1/drafts/{new_draft_id}', {'draft': orjson.dumps(draft_dict).decode()})
        self.assert_json_success(resp)
        new_draft = Draft.objects.get(id=new_draft_id, user_profile=hamlet)
        new_draft_dict = new_draft.to_dict()
        new_draft_dict.pop('id')
        self.assertEqual(new_draft_dict, draft_dict)

    def test_edit_non_existent_draft(self) -> None:
        if False:
            return 10
        hamlet = self.example_user('hamlet')
        initial_count = Draft.objects.count()
        draft_dict = {'type': 'stream', 'to': [10], 'topic': 'drafts', 'content': 'The API should be good', 'timestamp': 1595505700}
        resp = self.api_patch(hamlet, '/api/v1/drafts/999999999', {'draft': orjson.dumps(draft_dict).decode()})
        self.assert_json_error(resp, 'Draft does not exist', status_code=404)
        self.assertEqual(Draft.objects.count() - initial_count, 0)

    def test_edit_unowned_draft(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        hamlet = self.example_user('hamlet')
        visible_streams = self.get_streams(hamlet)
        stream_id = self.get_stream_id(visible_streams[0])
        draft_dict = {'type': 'stream', 'to': [stream_id], 'topic': 'drafts', 'content': 'The API should be good', 'timestamp': 1595505700}
        resp = self.api_post(hamlet, '/api/v1/drafts', {'drafts': orjson.dumps([draft_dict]).decode()})
        self.assert_json_success(resp)
        new_draft_id = orjson.loads(resp.content)['ids'][0]
        modified_draft_dict = deepcopy(draft_dict)
        modified_draft_dict['content'] = '???'
        zoe = self.example_user('ZOE')
        resp = self.api_patch(zoe, f'/api/v1/drafts/{new_draft_id}', {'draft': orjson.dumps(draft_dict).decode()})
        self.assert_json_error(resp, 'Draft does not exist', status_code=404)
        existing_draft = Draft.objects.get(id=new_draft_id, user_profile=hamlet)
        existing_draft_dict = existing_draft.to_dict()
        existing_draft_dict.pop('id')
        self.assertEqual(existing_draft_dict, draft_dict)

class DraftDeleteTests(ZulipTestCase):

    def test_require_enable_drafts_synchronization(self) -> None:
        if False:
            while True:
                i = 10
        hamlet = self.example_user('hamlet')
        hamlet.enable_drafts_synchronization = False
        hamlet.save()
        resp = self.api_delete(hamlet, '/api/v1/drafts/1')
        self.assert_json_error(resp, 'User has disabled synchronizing drafts.')

    def test_delete_draft_successfully(self) -> None:
        if False:
            i = 10
            return i + 15
        hamlet = self.example_user('hamlet')
        visible_streams = self.get_streams(hamlet)
        stream_id = self.get_stream_id(visible_streams[0])
        initial_count = Draft.objects.count()
        draft_dict = {'type': 'stream', 'to': [stream_id], 'topic': 'drafts', 'content': 'The API should be good', 'timestamp': 1595505700}
        resp = self.api_post(hamlet, '/api/v1/drafts', {'drafts': orjson.dumps([draft_dict]).decode()})
        self.assert_json_success(resp)
        new_draft_id = orjson.loads(resp.content)['ids'][0]
        self.assertEqual(Draft.objects.count() - initial_count, 1)
        resp = self.api_delete(hamlet, f'/api/v1/drafts/{new_draft_id}')
        self.assert_json_success(resp)
        self.assertEqual(Draft.objects.count() - initial_count, 0)

    def test_delete_non_existent_draft(self) -> None:
        if False:
            print('Hello World!')
        hamlet = self.example_user('hamlet')
        initial_count = Draft.objects.count()
        resp = self.api_delete(hamlet, '/api/v1/drafts/9999999999')
        self.assert_json_error(resp, 'Draft does not exist', status_code=404)
        self.assertEqual(Draft.objects.count() - initial_count, 0)

    def test_delete_unowned_draft(self) -> None:
        if False:
            while True:
                i = 10
        hamlet = self.example_user('hamlet')
        visible_streams = self.get_streams(hamlet)
        stream_id = self.get_stream_id(visible_streams[0])
        initial_count = Draft.objects.count()
        draft_dict = {'type': 'stream', 'to': [stream_id], 'topic': 'drafts', 'content': 'The API should be good', 'timestamp': 1595505700}
        resp = self.api_post(hamlet, '/api/v1/drafts', {'drafts': orjson.dumps([draft_dict]).decode()})
        self.assert_json_success(resp)
        new_draft_id = orjson.loads(resp.content)['ids'][0]
        zoe = self.example_user('ZOE')
        resp = self.api_delete(zoe, f'/api/v1/drafts/{new_draft_id}')
        self.assert_json_error(resp, 'Draft does not exist', status_code=404)
        self.assertEqual(Draft.objects.count() - initial_count, 1)
        existing_draft = Draft.objects.get(id=new_draft_id, user_profile=hamlet)
        existing_draft_dict = existing_draft.to_dict()
        existing_draft_dict.pop('id')
        self.assertEqual(existing_draft_dict, draft_dict)

class DraftFetchTest(ZulipTestCase):

    def test_require_enable_drafts_synchronization(self) -> None:
        if False:
            print('Hello World!')
        hamlet = self.example_user('hamlet')
        hamlet.enable_drafts_synchronization = False
        hamlet.save()
        resp = self.api_get(hamlet, '/api/v1/drafts')
        self.assert_json_error(resp, 'User has disabled synchronizing drafts.')

    def test_fetch_drafts(self) -> None:
        if False:
            return 10
        initial_count = Draft.objects.count()
        hamlet = self.example_user('hamlet')
        zoe = self.example_user('ZOE')
        othello = self.example_user('othello')
        visible_stream_id = self.get_stream_id(self.get_streams(hamlet)[0])
        draft_dicts = [{'type': 'stream', 'to': [visible_stream_id], 'topic': 'thinking out loud', 'content': 'What if pigs really could fly?', 'timestamp': 15954790197}, {'type': 'private', 'to': [zoe.id, othello.id], 'topic': '', 'content': 'What if made it possible to sync drafts in Zulip?', 'timestamp': 15954790198}, {'type': 'private', 'to': [zoe.id], 'topic': '', 'content': 'What if made it possible to sync drafts in Zulip?', 'timestamp': 15954790199}]
        payload = {'drafts': orjson.dumps(draft_dicts).decode()}
        resp = self.api_post(hamlet, '/api/v1/drafts', payload)
        self.assert_json_success(resp)
        self.assertEqual(Draft.objects.count() - initial_count, 3)
        zoe_draft_dicts = [{'type': 'private', 'to': [hamlet.id], 'topic': '', 'content': 'Hello there!', 'timestamp': 15954790200}]
        payload = {'drafts': orjson.dumps(zoe_draft_dicts).decode()}
        resp = self.api_post(zoe, '/api/v1/drafts', payload)
        self.assert_json_success(resp)
        self.assertEqual(Draft.objects.count() - initial_count, 4)
        resp = self.api_get(hamlet, '/api/v1/drafts')
        self.assert_json_success(resp)
        data = orjson.loads(resp.content)
        self.assertEqual(data['count'], 3)
        first_draft_id = Draft.objects.filter(user_profile=hamlet).order_by('id')[0].id
        expected_draft_contents: List[Dict[str, object]] = [{'id': first_draft_id + i, **draft_dicts[i]} for i in range(3)]
        self.assertEqual(data['drafts'], expected_draft_contents)