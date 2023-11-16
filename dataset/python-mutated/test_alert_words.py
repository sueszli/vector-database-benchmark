import orjson
from zerver.actions.alert_words import do_add_alert_words, do_remove_alert_words
from zerver.lib.alert_words import alert_words_in_realm, user_alert_words
from zerver.lib.test_classes import ZulipTestCase
from zerver.lib.test_helpers import most_recent_message, most_recent_usermessage
from zerver.models import AlertWord, UserProfile

class AlertWordTests(ZulipTestCase):
    interesting_alert_word_list = ['alert', 'multi-word word', '☃']

    def get_user(self) -> UserProfile:
        if False:
            return 10
        user = self.example_user('hamlet')
        AlertWord.objects.filter(user_profile=user).delete()
        return user

    def test_internal_endpoint(self) -> None:
        if False:
            while True:
                i = 10
        user = self.get_user()
        self.login_user(user)
        params = {'alert_words': orjson.dumps(['milk', 'cookies']).decode()}
        result = self.client_post('/json/users/me/alert_words', params)
        self.assert_json_success(result)
        words = user_alert_words(user)
        self.assertEqual(set(words), {'milk', 'cookies'})

    def test_default_no_words(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Users start out with no alert words.\n        '
        user = self.get_user()
        words = user_alert_words(user)
        self.assertEqual(words, [])

    def test_basics(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Verifies the basic behavior of modifying alert words.\n\n        Also verifies the cache-flushing behavior.\n        '
        user = self.get_user()
        realm_alert_words = alert_words_in_realm(user.realm)
        self.assert_length(realm_alert_words.get(user.id, []), 0)
        do_add_alert_words(user, self.interesting_alert_word_list)
        words = user_alert_words(user)
        self.assertEqual(set(words), set(self.interesting_alert_word_list))
        realm_alert_words = alert_words_in_realm(user.realm)
        self.assert_length(realm_alert_words[user.id], 3)
        do_add_alert_words(user, {'ALert', 'ALERT'})
        words = user_alert_words(user)
        self.assertEqual(set(words), set(self.interesting_alert_word_list))
        realm_alert_words = alert_words_in_realm(user.realm)
        self.assert_length(realm_alert_words[user.id], 3)
        do_remove_alert_words(user, {'ALert'})
        words = user_alert_words(user)
        self.assertEqual(set(words), set(self.interesting_alert_word_list) - {'alert'})
        realm_alert_words = alert_words_in_realm(user.realm)
        self.assert_length(realm_alert_words[user.id], 2)

    def test_remove_word(self) -> None:
        if False:
            print('Hello World!')
        '\n        Removing alert words works via do_remove_alert_words, even\n        for multi-word and non-ascii words.\n        '
        user = self.get_user()
        expected_remaining_alerts = set(self.interesting_alert_word_list)
        do_add_alert_words(user, self.interesting_alert_word_list)
        for alert_word in self.interesting_alert_word_list:
            do_remove_alert_words(user, [alert_word])
            expected_remaining_alerts.remove(alert_word)
            actual_remaining_alerts = user_alert_words(user)
            self.assertEqual(set(actual_remaining_alerts), expected_remaining_alerts)

    def test_realm_words(self) -> None:
        if False:
            while True:
                i = 10
        '\n        We can gather alert words for an entire realm via\n        alert_words_in_realm. Alerts added for one user do not impact other\n        users.\n        '
        AlertWord.objects.all().delete()
        user1 = self.get_user()
        do_add_alert_words(user1, self.interesting_alert_word_list)
        user2 = self.example_user('othello')
        do_add_alert_words(user2, ['another'])
        realm_words = alert_words_in_realm(user2.realm)
        self.assert_length(realm_words, 2)
        self.assertEqual(set(realm_words.keys()), {user1.id, user2.id})
        self.assertEqual(set(realm_words[user1.id]), set(self.interesting_alert_word_list))
        self.assertEqual(set(realm_words[user2.id]), {'another'})

    def test_json_list_default(self) -> None:
        if False:
            print('Hello World!')
        user = self.get_user()
        self.login_user(user)
        result = self.client_get('/json/users/me/alert_words')
        response_dict = self.assert_json_success(result)
        self.assertEqual(response_dict['alert_words'], [])

    def test_json_list_nonempty(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user = self.get_user()
        do_add_alert_words(user, ['one', 'two', 'three'])
        self.login_user(user)
        result = self.client_get('/json/users/me/alert_words')
        response_dict = self.assert_json_success(result)
        self.assertEqual(set(response_dict['alert_words']), {'one', 'two', 'three'})

    def test_json_list_add(self) -> None:
        if False:
            i = 10
            return i + 15
        user = self.get_user()
        self.login_user(user)
        result = self.client_post('/json/users/me/alert_words', {'alert_words': orjson.dumps(['one ', '\n two', 'three']).decode()})
        response_dict = self.assert_json_success(result)
        self.assertEqual(set(response_dict['alert_words']), {'one', 'two', 'three'})
        result = self.client_post('/json/users/me/alert_words', {'alert_words': orjson.dumps(['long' * 26]).decode()})
        self.assert_json_error(result, 'alert_words[0] is too long (limit: 100 characters)')

    def test_json_list_remove(self) -> None:
        if False:
            print('Hello World!')
        user = self.get_user()
        self.login_user(user)
        result = self.client_post('/json/users/me/alert_words', {'alert_words': orjson.dumps(['one', 'two', 'three']).decode()})
        response_dict = self.assert_json_success(result)
        self.assertEqual(set(response_dict['alert_words']), {'one', 'two', 'three'})
        result = self.client_delete('/json/users/me/alert_words', {'alert_words': orjson.dumps(['one']).decode()})
        response_dict = self.assert_json_success(result)
        self.assertEqual(set(response_dict['alert_words']), {'two', 'three'})

    def message_does_alert(self, user: UserProfile, message: str) -> bool:
        if False:
            return 10
        'Send a bunch of messages as othello, so our user is notified'
        self.send_stream_message(self.example_user('othello'), 'Denmark', message)
        user_message = most_recent_usermessage(user)
        return 'has_alert_word' in user_message.flags_list()

    def test_alert_flags(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user = self.get_user()
        self.login_user(user)
        result = self.client_post('/json/users/me/alert_words', {'alert_words': orjson.dumps(['one', 'two', 'three']).decode()})
        response_dict = self.assert_json_success(result)
        self.assertEqual(set(response_dict['alert_words']), {'one', 'two', 'three'})
        self.assertTrue(self.message_does_alert(user, 'Normal alert one time'))
        self.assertTrue(self.message_does_alert(user, 'Normal alert one'))
        self.assertTrue(self.message_does_alert(user, 'two normal alerts'))
        self.assertTrue(self.message_does_alert(user, 'This one? should alert'))
        self.assertTrue(self.message_does_alert(user, 'Definitely time for three.'))
        self.assertTrue(self.message_does_alert(user, "One two three o'clock"))
        self.assertTrue(self.message_does_alert(user, "One o'clock"))
        self.assertTrue(self.message_does_alert(user, "Case of ONE, won't stop me"))
        self.assertFalse(self.message_does_alert(user, "Don't alert on http://t.co/one/ URLs"))
        self.assertFalse(self.message_does_alert(user, "Don't alert on http://t.co/one URLs"))
        self.assertFalse(self.message_does_alert(user, "Don't alert on clone, twofold or seventytwofold"))

    def test_update_alert_words(self) -> None:
        if False:
            return 10
        user = self.get_user()
        self.login_user(user)
        result = self.client_post('/json/users/me/alert_words', {'alert_words': orjson.dumps(['ALERT']).decode()})
        content = 'this is an ALERT for you'
        self.send_stream_message(user, 'Denmark', content)
        self.assert_json_success(result)
        original_message = most_recent_message(user)
        user_message = most_recent_usermessage(user)
        self.assertIn('has_alert_word', user_message.flags_list())
        result = self.client_patch('/json/messages/' + str(original_message.id), {'content': 'new ALERT for you'})
        self.assert_json_success(result)
        user_message = most_recent_usermessage(user)
        self.assertEqual(user_message.message.content, 'new ALERT for you')
        self.assertIn('has_alert_word', user_message.flags_list())
        result = self.client_patch('/json/messages/' + str(original_message.id), {'content': 'sorry false alarm'})
        self.assert_json_success(result)
        user_message = most_recent_usermessage(user)
        self.assertEqual(user_message.message.content, 'sorry false alarm')
        self.assertNotIn('has_alert_word', user_message.flags_list())