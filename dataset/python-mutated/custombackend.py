"""A custom backend for testing."""
from django.core.mail.backends.base import BaseEmailBackend

class EmailBackend(BaseEmailBackend):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.test_outbox = []

    def send_messages(self, email_messages):
        if False:
            i = 10
            return i + 15
        self.test_outbox.extend(email_messages)
        return len(email_messages)