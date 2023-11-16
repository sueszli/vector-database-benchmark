import textwrap
from twisted.trial import unittest
from buildbot.util import raml

class TestRaml(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.api = raml.RamlSpec()

    def test_api(self):
        if False:
            print('Hello World!')
        self.assertTrue(self.api.api is not None)

    def test_endpoints(self):
        if False:
            i = 10
            return i + 15
        self.assertIn('/masters/{masterid}/builders/{builderid}/workers/{workerid}', self.api.endpoints.keys())

    def test_endpoints_uri_parameters(self):
        if False:
            while True:
                i = 10
        self.assertEqual(str(self.api.endpoints['/masters/{masterid}/builders/{builderid}/workers/{workerid}']['uriParameters']), str(raml.OrderedDict([('masterid', raml.OrderedDict([('type', 'number'), ('description', 'the id of the master')])), ('builderid', raml.OrderedDict([('type', 'number'), ('description', 'the id of the builder')])), ('workerid', raml.OrderedDict([('type', 'number'), ('description', 'the id of the worker')]))])))

    def test_types(self):
        if False:
            i = 10
            return i + 15
        self.assertIn('log', self.api.types.keys())

    def test_json_example(self):
        if False:
            return 10
        self.assertEqual(textwrap.dedent(self.api.format_json(self.api.types['build']['example'], 0)), textwrap.dedent('\n            {\n                "builderid": 10,\n                "buildid": 100,\n                "buildrequestid": 13,\n                "workerid": 20,\n                "complete": false,\n                "complete_at": null,\n                "masterid": 824,\n                "number": 1,\n                "results": null,\n                "started_at": 1451001600,\n                "state_string": "created",\n                "properties": {}\n            }').strip())

    def test_endpoints_by_type(self):
        if False:
            print('Hello World!')
        self.assertIn('/masters/{masterid}/builders/{builderid}/workers/{workerid}', self.api.endpoints_by_type['worker'].keys())

    def test_iter_actions(self):
        if False:
            return 10
        build = self.api.endpoints_by_type['build']
        actions = dict(self.api.iter_actions(build['/builds/{buildid}']))
        self.assertEqual(sorted(actions.keys()), sorted(['rebuild', 'stop']))

    def test_rawendpoints(self):
        if False:
            i = 10
            return i + 15
        self.assertIn('/steps/{stepid}/logs/{log_slug}/raw', self.api.rawendpoints.keys())