from oslo_config import cfg
from st2common.util.monkey_patch import use_select_poll_workaround
use_select_poll_workaround()
from st2common.content.bootstrap import register_content
from st2common.persistence.pack import Pack
from st2common.persistence.pack import Config
from st2common.persistence.pack import ConfigSchema
from st2common.persistence.actionalias import ActionAlias
from st2common.persistence.action import Action
from st2common.persistence.rule import Rule
from st2common.persistence.policy import Policy
from st2common.persistence.sensor import SensorType as Sensor
from st2common.persistence.trigger import TriggerType
from st2tests.base import BaseActionTestCase
from st2tests.base import CleanDbTestCase
from st2tests.fixtures.packs.dummy_pack_1.fixture import PACK_NAME as DUMMY_PACK_1, PACK_PATH as PACK_PATH_1
from pack_mgmt.unload import UnregisterPackAction
__all__ = ['UnloadActionTestCase']

class UnloadActionTestCase(BaseActionTestCase, CleanDbTestCase):
    action_cls = UnregisterPackAction

    def setUp(self):
        if False:
            return 10
        super(UnloadActionTestCase, self).setUp()
        pack_dbs = Pack.get_all()
        config_schema_dbs = ConfigSchema.get_all()
        config_dbs = Config.get_all()
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(config_schema_dbs), 0)
        self.assertEqual(len(config_dbs), 0)
        cfg.CONF.set_override(name='all', override=True, group='register')
        cfg.CONF.set_override(name='pack', override=PACK_PATH_1, group='register')
        cfg.CONF.set_override(name='no_fail_on_failure', override=True, group='register')
        register_content()

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        pack = DUMMY_PACK_1
        pack_dbs = Pack.query(ref=pack)
        action_dbs = Action.query(pack=pack)
        alias_dbs = ActionAlias.query(pack=pack)
        rule_dbs = Rule.query(pack=pack)
        sensor_dbs = Sensor.query(pack=pack)
        trigger_type_dbs = TriggerType.query(pack=pack)
        policy_dbs = Policy.query(pack=pack)
        config_schema_dbs = ConfigSchema.query(pack=pack)
        config_dbs = Config.query(pack=pack)
        self.assertEqual(len(pack_dbs), 1)
        self.assertEqual(len(action_dbs), 1)
        self.assertEqual(len(alias_dbs), 3)
        self.assertEqual(len(rule_dbs), 1)
        self.assertEqual(len(sensor_dbs), 3)
        self.assertEqual(len(trigger_type_dbs), 4)
        self.assertEqual(len(policy_dbs), 2)
        self.assertEqual(len(config_schema_dbs), 1)
        self.assertEqual(len(config_dbs), 1)
        action = self.get_action_instance()
        action.run(packs=[pack])
        pack_dbs = Pack.query(ref=pack)
        action_dbs = Action.query(pack=pack)
        alias_dbs = ActionAlias.query(pack=pack)
        rule_dbs = Rule.query(pack=pack)
        sensor_dbs = Sensor.query(pack=pack)
        trigger_type_dbs = TriggerType.query(pack=pack)
        policy_dbs = Policy.query(pack=pack)
        config_schema_dbs = ConfigSchema.query(pack=pack)
        config_dbs = Config.query(pack=pack)
        self.assertEqual(len(pack_dbs), 0)
        self.assertEqual(len(action_dbs), 0)
        self.assertEqual(len(alias_dbs), 0)
        self.assertEqual(len(rule_dbs), 0)
        self.assertEqual(len(sensor_dbs), 0)
        self.assertEqual(len(trigger_type_dbs), 0)
        self.assertEqual(len(policy_dbs), 0)
        self.assertEqual(len(config_schema_dbs), 0)
        self.assertEqual(len(config_dbs), 0)