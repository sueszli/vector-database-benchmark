"""
.. module: security_monkey.tests.views.test_watcher_config
    :platform: Unix

.. version:: $$VERSION$$
.. moduleauthor:: Bridgewater OSS <opensource@bwater.com>


"""
from security_monkey.tests.views import SecurityMonkeyApiTestCase
from security_monkey.watcher import watcher_registry
from security_monkey.datastore import WatcherConfig
from security_monkey import db
from mock import patch
import json

class MockWatcher(object):

    def __init__(self, accounts=None, debug=False):
        if False:
            while True:
                i = 10
        self.accounts = accounts
watcher_configs = [{'type': 'MockWatcher1', 'index': 'index1', 'interval': 1440}, {'type': 'MockWatcher2', 'index': 'index2', 'interval': 1440}, {'type': 'MockWatcher3', 'index': 'index3', 'interval': 1440}]
test_watcher_registry = {}
for config in watcher_configs:
    watcher = type(config['type'], (MockWatcher,), {'index': config['index'], 'interval': config['interval']})
    test_watcher_registry[config['index']] = watcher

@patch.dict(watcher_registry, test_watcher_registry, clear=True)
class WatcherConfigApiTestCase(SecurityMonkeyApiTestCase):

    def test_get_empty_watcher_configs(self):
        if False:
            for i in range(10):
                print('nop')
        r = self.test_app.get('/api/1/watcher_config', headers=self.headers)
        r_json = json.loads(r.data)
        assert r.status_code == 200
        assert len(r_json['items']) == len(watcher_configs)
        assert r_json['items'][0]['id'] == 0

    def test_get_watcher_configs(self):
        if False:
            i = 10
            return i + 15
        watcher_config = WatcherConfig(index='index1', interval=1440, active=True)
        db.session.add(watcher_config)
        db.session.commit()
        db.session.refresh(watcher_config)
        r = self.test_app.get('/api/1/watcher_config', headers=self.headers)
        r_json = json.loads(r.data)
        assert r.status_code == 200
        assert len(r_json['items']) == len(watcher_configs)
        assert r_json['items'][0]['id'] != 0

    def test_put_watcher_config(self):
        if False:
            i = 10
            return i + 15
        watcher_config = WatcherConfig(index='index1', interval=1440, active=True)
        db.session.add(watcher_config)
        db.session.commit()
        db.session.refresh(watcher_config)
        d = dict(index='account', interval=1440, active=True)
        r = self.test_app.put('/api/1/watcher_config/{}'.format(watcher_config.id), headers=self.headers, data=json.dumps(d))
        assert r.status_code == 200

    def test_put_watcher_config_wrong_id(self):
        if False:
            i = 10
            return i + 15
        watcher_config = WatcherConfig(index='index1', interval=1440, active=True)
        db.session.add(watcher_config)
        db.session.commit()
        db.session.refresh(watcher_config)
        d = dict(index='account', interval=1440, active=True)
        r = self.test_app.put('/api/1/watcher_config/{}'.format('100'), headers=self.headers, data=json.dumps(d))
        assert r.status_code == 500

    def test_put_watcher_config_wrong_data(self):
        if False:
            i = 10
            return i + 15
        watcher_config = WatcherConfig(index='index1', interval=1440, active=True)
        db.session.add(watcher_config)
        db.session.commit()
        db.session.refresh(watcher_config)
        d = dict(index='account', foobar=1440, active=True)
        r = self.test_app.put('/api/1/watcher_config/{}'.format(watcher_config.id), headers=self.headers, data=json.dumps(d))
        assert r.status_code == 400