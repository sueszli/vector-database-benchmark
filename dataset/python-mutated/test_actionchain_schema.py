from __future__ import absolute_import
import unittest2
from jsonschema.exceptions import ValidationError
from st2common.models.system import actionchain
from st2tests.fixtures.generic.fixture import PACK_NAME as FIXTURES_PACK
from st2tests.fixturesloader import FixturesLoader
TEST_FIXTURES = {'actionchains': ['chain1.yaml', 'malformedchain.yaml', 'no_default_chain.yaml', 'chain_with_vars.yaml', 'chain_with_publish.yaml']}
FIXTURES = FixturesLoader().load_fixtures(fixtures_pack=FIXTURES_PACK, fixtures_dict=TEST_FIXTURES)
CHAIN_1 = FIXTURES['actionchains']['chain1.yaml']
MALFORMED_CHAIN = FIXTURES['actionchains']['malformedchain.yaml']
NO_DEFAULT_CHAIN = FIXTURES['actionchains']['no_default_chain.yaml']
CHAIN_WITH_VARS = FIXTURES['actionchains']['chain_with_vars.yaml']
CHAIN_WITH_PUBLISH = FIXTURES['actionchains']['chain_with_publish.yaml']

class ActionChainSchemaTest(unittest2.TestCase):

    def test_actionchain_schema_valid(self):
        if False:
            return 10
        chain = actionchain.ActionChain(**CHAIN_1)
        self.assertEqual(len(chain.chain), len(CHAIN_1['chain']))
        self.assertEqual(chain.default, CHAIN_1['default'])

    def test_actionchain_no_default(self):
        if False:
            return 10
        chain = actionchain.ActionChain(**NO_DEFAULT_CHAIN)
        self.assertEqual(len(chain.chain), len(NO_DEFAULT_CHAIN['chain']))
        self.assertEqual(chain.default, None)

    def test_actionchain_with_vars(self):
        if False:
            i = 10
            return i + 15
        chain = actionchain.ActionChain(**CHAIN_WITH_VARS)
        self.assertEqual(len(chain.chain), len(CHAIN_WITH_VARS['chain']))
        self.assertEqual(len(chain.vars), len(CHAIN_WITH_VARS['vars']))

    def test_actionchain_with_publish(self):
        if False:
            for i in range(10):
                print('nop')
        chain = actionchain.ActionChain(**CHAIN_WITH_PUBLISH)
        self.assertEqual(len(chain.chain), len(CHAIN_WITH_PUBLISH['chain']))
        self.assertEqual(len(chain.chain[0].publish), len(CHAIN_WITH_PUBLISH['chain'][0]['publish']))

    def test_actionchain_schema_invalid(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValidationError):
            actionchain.ActionChain(**MALFORMED_CHAIN)