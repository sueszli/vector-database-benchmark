from __future__ import absolute_import
from st2common.persistence.trigger import Trigger
from st2common.triggers import register_internal_trigger_types
from st2tests.base import DbTestCase

class TestRegisterInternalTriggers(DbTestCase):

    def test_register_internal_trigger_types(self):
        if False:
            for i in range(10):
                print('nop')
        registered_trigger_types_db = register_internal_trigger_types()
        for trigger_type_db in registered_trigger_types_db:
            self._validate_shadow_trigger(trigger_type_db)

    def _validate_shadow_trigger(self, trigger_type_db):
        if False:
            return 10
        if trigger_type_db.parameters_schema:
            return
        trigger_type_ref = trigger_type_db.get_reference().ref
        triggers = Trigger.query(type=trigger_type_ref)
        self.assertTrue(len(triggers) > 0, 'Shadow trigger not created for %s.' % trigger_type_ref)