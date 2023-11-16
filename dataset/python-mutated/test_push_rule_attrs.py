import synapse
from synapse.api.errors import Codes
from synapse.rest.client import login, push_rule, room
from tests.unittest import HomeserverTestCase

class PushRuleAttributesTestCase(HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, room.register_servlets, login.register_servlets, push_rule.register_servlets]
    hijack_auth = False

    def test_enabled_on_creation(self) -> None:
        if False:
            return 10
        "\n        Tests the GET and PUT of push rules' `enabled` endpoints.\n        Tests that a rule is enabled upon creation, even though a rule with that\n            ruleId existed previously and was disabled.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        body = {'conditions': [{'kind': 'event_match', 'key': 'sender', 'pattern': '@user2:hs'}], 'actions': ['notify', {'set_tweak': 'highlight'}]}
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend', body, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['enabled'], True)

    def test_enabled_on_recreation(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Tests the GET and PUT of push rules' `enabled` endpoints.\n        Tests that a rule is enabled upon creation, even if a rule with that\n            ruleId existed previously and was disabled.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        body = {'conditions': [{'kind': 'event_match', 'key': 'sender', 'pattern': '@user2:hs'}], 'actions': ['notify', {'set_tweak': 'highlight'}]}
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend', body, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend/enabled', {'enabled': False}, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['enabled'], False)
        channel = self.make_request('DELETE', '/pushrules/global/override/best.friend', access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend', body, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['enabled'], True)

    def test_enabled_disable(self) -> None:
        if False:
            return 10
        "\n        Tests the GET and PUT of push rules' `enabled` endpoints.\n        Tests that a rule is disabled and enabled when we ask for it.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        body = {'conditions': [{'kind': 'event_match', 'key': 'sender', 'pattern': '@user2:hs'}], 'actions': ['notify', {'set_tweak': 'highlight'}]}
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend', body, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend/enabled', {'enabled': False}, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['enabled'], False)
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend/enabled', {'enabled': True}, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['enabled'], True)

    def test_enabled_404_when_get_non_existent(self) -> None:
        if False:
            print('Hello World!')
        "\n        Tests that `enabled` gives 404 when the rule doesn't exist.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        body = {'conditions': [{'kind': 'event_match', 'key': 'sender', 'pattern': '@user2:hs'}], 'actions': ['notify', {'set_tweak': 'highlight'}]}
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend', body, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('DELETE', '/pushrules/global/override/best.friend', access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_enabled_404_when_get_non_existent_server_rule(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Tests that `enabled` gives 404 when the server-default rule doesn't exist.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        channel = self.make_request('GET', '/pushrules/global/override/.m.muahahaha/enabled', access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_enabled_404_when_put_non_existent_rule(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Tests that `enabled` gives 404 when we put to a rule that doesn't exist.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend/enabled', {'enabled': True}, access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_enabled_404_when_put_non_existent_server_rule(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Tests that `enabled` gives 404 when we put to a server-default rule that doesn't exist.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        channel = self.make_request('PUT', '/pushrules/global/override/.m.muahahah/enabled', {'enabled': True}, access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_actions_get(self) -> None:
        if False:
            return 10
        '\n        Tests that `actions` gives you what you expect on a fresh rule.\n        '
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        body = {'conditions': [{'kind': 'event_match', 'key': 'sender', 'pattern': '@user2:hs'}], 'actions': ['notify', {'set_tweak': 'highlight'}]}
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend', body, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/actions', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['actions'], ['notify', {'set_tweak': 'highlight'}])

    def test_actions_put(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Tests that PUT on actions updates the value you'd get from GET.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        body = {'conditions': [{'kind': 'event_match', 'key': 'sender', 'pattern': '@user2:hs'}], 'actions': ['notify', {'set_tweak': 'highlight'}]}
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend', body, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend/actions', {'actions': ['dont_notify']}, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/actions', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['actions'], ['dont_notify'])

    def test_actions_404_when_get_non_existent(self) -> None:
        if False:
            return 10
        "\n        Tests that `actions` gives 404 when the rule doesn't exist.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        body = {'conditions': [{'kind': 'event_match', 'key': 'sender', 'pattern': '@user2:hs'}], 'actions': ['notify', {'set_tweak': 'highlight'}]}
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend', body, access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('DELETE', '/pushrules/global/override/best.friend', access_token=token)
        self.assertEqual(channel.code, 200)
        channel = self.make_request('GET', '/pushrules/global/override/best.friend/enabled', access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_actions_404_when_get_non_existent_server_rule(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Tests that `actions` gives 404 when the server-default rule doesn't exist.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        channel = self.make_request('GET', '/pushrules/global/override/.m.muahahaha/actions', access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_actions_404_when_put_non_existent_rule(self) -> None:
        if False:
            print('Hello World!')
        "\n        Tests that `actions` gives 404 when putting to a rule that doesn't exist.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        channel = self.make_request('PUT', '/pushrules/global/override/best.friend/actions', {'actions': ['dont_notify']}, access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_actions_404_when_put_non_existent_server_rule(self) -> None:
        if False:
            while True:
                i = 10
        "\n        Tests that `actions` gives 404 when putting to a server-default rule that doesn't exist.\n        "
        self.register_user('user', 'pass')
        token = self.login('user', 'pass')
        channel = self.make_request('PUT', '/pushrules/global/override/.m.muahahah/actions', {'actions': ['dont_notify']}, access_token=token)
        self.assertEqual(channel.code, 404)
        self.assertEqual(channel.json_body['errcode'], Codes.NOT_FOUND)

    def test_contains_user_name(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Tests that `contains_user_name` rule is present and have proper value in `pattern`.\n        '
        username = 'bob'
        self.register_user(username, 'pass')
        token = self.login(username, 'pass')
        channel = self.make_request('GET', '/pushrules/global/content/.m.rule.contains_user_name', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual({'rule_id': '.m.rule.contains_user_name', 'default': True, 'enabled': True, 'pattern': username, 'actions': ['notify', {'set_tweak': 'highlight'}, {'set_tweak': 'sound', 'value': 'default'}]}, channel.json_body)

    def test_is_user_mention(self) -> None:
        if False:
            return 10
        '\n        Tests that `is_user_mention` rule is present and have proper value in `value`.\n        '
        user = self.register_user('bob', 'pass')
        token = self.login('bob', 'pass')
        channel = self.make_request('GET', '/pushrules/global/override/.m.rule.is_user_mention', access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual({'rule_id': '.m.rule.is_user_mention', 'default': True, 'enabled': True, 'conditions': [{'kind': 'event_property_contains', 'key': 'content.m\\.mentions.user_ids', 'value': user}], 'actions': ['notify', {'set_tweak': 'highlight'}, {'set_tweak': 'sound', 'value': 'default'}]}, channel.json_body)