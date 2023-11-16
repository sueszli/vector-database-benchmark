"""
.. module: security_monkey.tests.test_account
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>

"""
from security_monkey.tests.views import SecurityMonkeyApiTestCase
from security_monkey.datastore import Account, AccountType, Technology, Item, ItemRevision
from security_monkey.tests import db
from security_monkey import ARN_PREFIX
from datetime import datetime, timedelta
import json

class ItemApiTestCase(SecurityMonkeyApiTestCase):

    def test_first_last_seen(self):
        if False:
            i = 10
            return i + 15
        self._setup_one_two_revisions()
        r = self.test_app.get('/api/1/items', headers=self.headers)
        assert r.status_code == 200
        r_json = json.loads(r.data)
        assert len(r_json['items']) == 1
        assert r_json['items'][0]['first_seen'] == '2016-11-02 00:00:00'
        assert r_json['items'][0]['last_seen'] == '2016-11-03 00:00:00'

    def _setup_one_two_revisions(self):
        if False:
            return 10
        account_type_result = AccountType.query.filter(AccountType.name == 'AWS').first()
        if not account_type_result:
            account_type_result = AccountType(name='AWS')
            db.session.add(account_type_result)
            db.session.commit()
        account = Account(identifier='012345678910', name='testing', account_type_id=account_type_result.id)
        technology = Technology(name='iamrole')
        item = Item(region='us-west-2', name='testrole', arn=ARN_PREFIX + ':iam::012345678910:role/testrole', technology=technology, account=account)
        self.now = datetime(2016, 11, 3)
        self.yesterday = self.now - timedelta(days=1)
        item.revisions.append(ItemRevision(active=True, config={}, date_created=self.now))
        item.revisions.append(ItemRevision(active=True, config={}, date_created=self.yesterday))
        db.session.add(account)
        db.session.add(technology)
        db.session.add(item)
        db.session.commit()
        items = Item.query.all()
        for item in items:
            latest_revision = item.revisions.first()
            item.latest_revision_id = latest_revision.id
            db.session.add(item)
            db.session.commit()