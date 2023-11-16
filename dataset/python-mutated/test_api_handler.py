import unittest
import json
from nyaa import api_handler, models
from tests import NyaaTestCase
from pprint import pprint

class ApiHandlerTests(NyaaTestCase):

    def test_no_authorization(self):
        if False:
            return 10
        " Test that API is locked unless you're logged in "
        rv = self.app.get('/api/info/1')
        data = json.loads(rv.get_data())
        self.assertDictEqual({'errors': ['Bad authorization']}, data)

    @unittest.skip('Not yet implemented')
    def test_bad_credentials(self):
        if False:
            while True:
                i = 10
        " Test that API is locked unless you're logged in "
        rv = self.app.get('/api/info/1')
        data = json.loads(rv.get_data())
        self.assertDictEqual({'errors': ['Bad authorization']}, data)
if __name__ == '__main__':
    unittest.main()