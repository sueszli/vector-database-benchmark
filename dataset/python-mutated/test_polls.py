from __future__ import annotations
from django.test import TestCase

class PollViewTests(TestCase):

    def test_index_view(self: PollViewTests) -> None:
        if False:
            i = 10
            return i + 15
        response = self.client.get('/')
        assert response.status_code == 200
        assert 'Hello, world' in str(response.content)