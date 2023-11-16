from oslo_config import cfg
from st2common.models.db import db_setup
from st2common.runners.base_action import Action as BaseAction
from st2common.persistence.pack import Pack
from st2common.persistence.pack import ConfigSchema
from st2common.persistence.pack import Config
from st2common.persistence.action import Action
from st2common.persistence.action import ActionAlias
from st2common.persistence.policy import Policy
from st2common.persistence.rule import Rule
from st2common.persistence.sensor import SensorType
from st2common.persistence.trigger import Trigger
from st2common.persistence.trigger import TriggerType
from st2common.constants.pack import SYSTEM_PACK_NAMES
from st2common.services.triggers import cleanup_trigger_db_for_rule
from st2common.exceptions.db import StackStormDBObjectNotFoundError
BLOCKED_PACKS = frozenset(SYSTEM_PACK_NAMES)

class UnregisterPackAction(BaseAction):

    def __init__(self, config=None, action_service=None):
        if False:
            i = 10
            return i + 15
        super(UnregisterPackAction, self).__init__(config=config, action_service=action_service)
        self.initialize()

    def initialize(self):
        if False:
            return 10
        username = cfg.CONF.database.username if hasattr(cfg.CONF.database, 'username') else None
        password = cfg.CONF.database.password if hasattr(cfg.CONF.database, 'password') else None
        db_setup(cfg.CONF.database.db_name, cfg.CONF.database.host, cfg.CONF.database.port, username=username, password=password, ssl=cfg.CONF.database.ssl, ssl_keyfile=cfg.CONF.database.ssl_keyfile, ssl_certfile=cfg.CONF.database.ssl_certfile, ssl_cert_reqs=cfg.CONF.database.ssl_cert_reqs, ssl_ca_certs=cfg.CONF.database.ssl_ca_certs, authentication_mechanism=cfg.CONF.database.authentication_mechanism, ssl_match_hostname=cfg.CONF.database.ssl_match_hostname)

    def run(self, packs):
        if False:
            print('Hello World!')
        intersection = BLOCKED_PACKS & frozenset(packs)
        if len(intersection) > 0:
            names = ', '.join(list(intersection))
            raise ValueError('Unregister includes an unregisterable pack - %s.' % names)
        for pack in packs:
            self.logger.debug('Removing pack %s.', pack)
            self._unregister_sensors(pack=pack)
            self._unregister_trigger_types(pack=pack)
            self._unregister_triggers(pack=pack)
            self._unregister_actions(pack=pack)
            self._unregister_rules(pack=pack)
            self._unregister_aliases(pack=pack)
            self._unregister_policies(pack=pack)
            self._unregister_pack(pack=pack)
            self.logger.info('Removed pack %s.', pack)

    def _unregister_sensors(self, pack):
        if False:
            for i in range(10):
                print('nop')
        return self._delete_pack_db_objects(pack=pack, access_cls=SensorType)

    def _unregister_trigger_types(self, pack):
        if False:
            i = 10
            return i + 15
        deleted_trigger_types_dbs = self._delete_pack_db_objects(pack=pack, access_cls=TriggerType)
        for trigger_type_db in deleted_trigger_types_dbs:
            rule_dbs = Rule.query(trigger=trigger_type_db.ref, pack__ne=trigger_type_db.pack)
            for rule_db in rule_dbs:
                self.logger.warning('Rule "%s" references deleted trigger "%s"' % (rule_db.name, trigger_type_db.ref))
        return deleted_trigger_types_dbs

    def _unregister_triggers(self, pack):
        if False:
            return 10
        return self._delete_pack_db_objects(pack=pack, access_cls=Trigger)

    def _unregister_actions(self, pack):
        if False:
            for i in range(10):
                print('nop')
        return self._delete_pack_db_objects(pack=pack, access_cls=Action)

    def _unregister_rules(self, pack):
        if False:
            for i in range(10):
                print('nop')
        deleted_rules = self._delete_pack_db_objects(pack=pack, access_cls=Rule)
        for rule_db in deleted_rules:
            cleanup_trigger_db_for_rule(rule_db=rule_db)
        return deleted_rules

    def _unregister_aliases(self, pack):
        if False:
            for i in range(10):
                print('nop')
        return self._delete_pack_db_objects(pack=pack, access_cls=ActionAlias)

    def _unregister_policies(self, pack):
        if False:
            print('Hello World!')
        return self._delete_pack_db_objects(pack=pack, access_cls=Policy)

    def _unregister_pack(self, pack):
        if False:
            while True:
                i = 10
        self._delete_pack_db_object(pack=pack)
        self._delete_config_schema_db_object(pack=pack)
        self._delete_pack_db_objects(pack=pack, access_cls=Config)
        return True

    def _delete_pack_db_object(self, pack):
        if False:
            i = 10
            return i + 15
        pack_db = None
        try:
            pack_db = Pack.get_by_ref(value=pack)
        except StackStormDBObjectNotFoundError:
            pack_db = None
        if not pack_db:
            try:
                pack_db = Pack.get_by_name(value=pack)
            except StackStormDBObjectNotFoundError:
                pack_db = None
        if not pack_db:
            self.logger.exception('Pack DB object not found')
            return
        try:
            Pack.delete(pack_db)
        except:
            self.logger.exception('Failed to remove DB object %s.', pack_db)

    def _delete_config_schema_db_object(self, pack):
        if False:
            for i in range(10):
                print('nop')
        try:
            config_schema_db = ConfigSchema.get_by_pack(value=pack)
        except StackStormDBObjectNotFoundError:
            self.logger.exception('ConfigSchemaDB object not found')
            return
        try:
            ConfigSchema.delete(config_schema_db)
        except:
            self.logger.exception('Failed to remove DB object %s.', config_schema_db)

    def _delete_pack_db_objects(self, pack, access_cls):
        if False:
            while True:
                i = 10
        db_objs = access_cls.get_all(pack=pack)
        deleted_objs = []
        for db_obj in db_objs:
            try:
                access_cls.delete(db_obj)
                deleted_objs.append(db_obj)
            except:
                self.logger.exception('Failed to remove DB object %s.', db_obj)
        return deleted_objs