from __future__ import absolute_import
import os
from st2common.bootstrap import aliasesregistrar
from st2common.persistence.action import ActionAlias
from st2tests import DbTestCase
from st2tests.fixtures.packs.dummy_pack_1.fixture import PACK_PATH as ALIASES_FIXTURE_PACK_PATH
__all__ = ['TestAliasRegistrar']
ALIASES_FIXTURE_PATH = os.path.join(ALIASES_FIXTURE_PACK_PATH, 'aliases')

class TestAliasRegistrar(DbTestCase):

    def test_alias_registration(self):
        if False:
            while True:
                i = 10
        (count, overridden) = aliasesregistrar.register_aliases(pack_dir=ALIASES_FIXTURE_PACK_PATH)
        self.assertEqual(count, len(os.listdir(ALIASES_FIXTURE_PATH)))
        self.assertEqual(0, overridden)
        action_alias_dbs = ActionAlias.get_all()
        self.assertEqual(action_alias_dbs[0].metadata_file, 'aliases/alias1.yaml')