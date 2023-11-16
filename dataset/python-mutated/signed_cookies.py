from django.contrib.sessions.backends.base import SessionBase
from django.core import signing

class SessionStore(SessionBase):

    def load(self):
        if False:
            return 10
        '\n        Load the data from the key itself instead of fetching from some\n        external data store. Opposite of _get_session_key(), raise BadSignature\n        if signature fails.\n        '
        try:
            return signing.loads(self.session_key, serializer=self.serializer, max_age=self.get_session_cookie_age(), salt='django.contrib.sessions.backends.signed_cookies')
        except Exception:
            self.create()
        return {}

    def create(self):
        if False:
            return 10
        '\n        To create a new key, set the modified flag so that the cookie is set\n        on the client for the current request.\n        '
        self.modified = True

    def save(self, must_create=False):
        if False:
            while True:
                i = 10
        '\n        To save, get the session key as a securely signed string and then set\n        the modified flag so that the cookie is set on the client for the\n        current request.\n        '
        self._session_key = self._get_session_key()
        self.modified = True

    def exists(self, session_key=None):
        if False:
            print('Hello World!')
        "\n        This method makes sense when you're talking to a shared resource, but\n        it doesn't matter when you're storing the information in the client's\n        cookie.\n        "
        return False

    def delete(self, session_key=None):
        if False:
            print('Hello World!')
        '\n        To delete, clear the session key and the underlying data structure\n        and set the modified flag so that the cookie is set on the client for\n        the current request.\n        '
        self._session_key = ''
        self._session_cache = {}
        self.modified = True

    def cycle_key(self):
        if False:
            while True:
                i = 10
        '\n        Keep the same data but with a new key. Call save() and it will\n        automatically save a cookie with a new key at the end of the request.\n        '
        self.save()

    def _get_session_key(self):
        if False:
            return 10
        '\n        Instead of generating a random string, generate a secure url-safe\n        base64-encoded string of data as our session key.\n        '
        return signing.dumps(self._session, compress=True, salt='django.contrib.sessions.backends.signed_cookies', serializer=self.serializer)

    @classmethod
    def clear_expired(cls):
        if False:
            return 10
        pass