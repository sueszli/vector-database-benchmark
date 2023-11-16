from __future__ import absolute_import
from datetime import timedelta
import bson
from st2common import log as logging
from st2common.garbage_collection.rule_enforcement import purge_rule_enforcements
from st2common.models.db.rule_enforcement import RuleEnforcementDB
from st2common.persistence.rule_enforcement import RuleEnforcement
from st2common.util import date as date_utils
from st2tests.base import CleanDbTestCase
LOG = logging.getLogger(__name__)

class TestPurgeRuleEnforcement(CleanDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        CleanDbTestCase.setUpClass()
        super(TestPurgeRuleEnforcement, cls).setUpClass()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestPurgeRuleEnforcement, self).setUp()

    def test_no_timestamp_doesnt_delete(self):
        if False:
            while True:
                i = 10
        now = date_utils.get_datetime_utc_now()
        TestPurgeRuleEnforcement._create_save_rule_enforcement(enforced_at=now - timedelta(days=20))
        self.assertEqual(len(RuleEnforcement.get_all()), 1)
        expected_msg = 'Specify a valid timestamp'
        self.assertRaisesRegexp(ValueError, expected_msg, purge_rule_enforcements, logger=LOG, timestamp=None)
        self.assertEqual(len(RuleEnforcement.get_all()), 1)

    def test_purge(self):
        if False:
            print('Hello World!')
        now = date_utils.get_datetime_utc_now()
        TestPurgeRuleEnforcement._create_save_rule_enforcement(enforced_at=now - timedelta(days=20))
        TestPurgeRuleEnforcement._create_save_rule_enforcement(enforced_at=now - timedelta(days=5))
        self.assertEqual(len(RuleEnforcement.get_all()), 2)
        purge_rule_enforcements(logger=LOG, timestamp=now - timedelta(days=10))
        self.assertEqual(len(RuleEnforcement.get_all()), 1)

    @staticmethod
    def _create_save_rule_enforcement(enforced_at):
        if False:
            while True:
                i = 10
        created = RuleEnforcementDB(trigger_instance_id=str(bson.ObjectId()), rule={'ref': 'foo_pack.foo_rule', 'uid': 'rule:foo_pack:foo_rule'}, execution_id=str(bson.ObjectId()), enforced_at=enforced_at)
        return RuleEnforcement.add_or_update(created)