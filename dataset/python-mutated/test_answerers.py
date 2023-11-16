from mock import Mock
from searx.answerers import answerers
from tests import SearxTestCase

class AnswererTest(SearxTestCase):

    def test_unicode_input(self):
        if False:
            print('Hello World!')
        query = Mock()
        unicode_payload = 'árvíztűrő tükörfúrógép'
        for answerer in answerers:
            query.query = '{} {}'.format(answerer.keywords[0], unicode_payload)
            self.assertTrue(isinstance(answerer.answer(query), list))