from __future__ import absolute_import
from datetime import timedelta
import bson
from st2common import log as logging
from st2common.garbage_collection.token import purge_tokens
from st2common.models.db.auth import TokenDB
from st2common.persistence.auth import Token
from st2common.util import date as date_utils
from st2tests.base import CleanDbTestCase
LOG = logging.getLogger(__name__)

class TestPurgeToken(CleanDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        CleanDbTestCase.setUpClass()
        super(TestPurgeToken, cls).setUpClass()

    def setUp(self):
        if False:
            return 10
        super(TestPurgeToken, self).setUp()

    def test_no_timestamp_doesnt_delete(self):
        if False:
            return 10
        now = date_utils.get_datetime_utc_now()
        TestPurgeToken._create_save_token(expiry_timestamp=now - timedelta(days=20))
        self.assertEqual(len(Token.get_all()), 1)
        expected_msg = 'Specify a valid timestamp'
        self.assertRaisesRegexp(ValueError, expected_msg, purge_tokens, logger=LOG, timestamp=None)
        self.assertEqual(len(Token.get_all()), 1)

    def test_purge(self):
        if False:
            while True:
                i = 10
        now = date_utils.get_datetime_utc_now()
        TestPurgeToken._create_save_token(expiry_timestamp=now - timedelta(days=20))
        TestPurgeToken._create_save_token(expiry_timestamp=now - timedelta(days=5))
        self.assertEqual(len(Token.get_all()), 2)
        purge_tokens(logger=LOG, timestamp=now - timedelta(days=10))
        self.assertEqual(len(Token.get_all()), 1)

    @staticmethod
    def _create_save_token(expiry_timestamp=None):
        if False:
            while True:
                i = 10
        created = TokenDB(id=str(bson.ObjectId()), user='pony', token=str(bson.ObjectId()), expiry=expiry_timestamp, metadata={'service': 'action-runner'})
        return Token.add_or_update(created)