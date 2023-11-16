from __future__ import absolute_import
from mongoengine.queryset import Q
from st2common import config
from st2common.script_setup import setup as common_setup
from st2common.script_setup import teardown as common_teardown
from st2common.persistence.rule import Rule
from st2common.persistence.trigger import Trigger
from st2common.models.db.trigger import TriggerDB

class TriggerMigrator(object):

    def _get_trigger_with_parameters(self):
        if False:
            while True:
                i = 10
        '\n        All TriggerDB that has a parameter.\n        '
        return TriggerDB.objects(Q(parameters__exists=True) & Q(parameters__nin=[{}]))

    def _get_rules_for_trigger(self, trigger_ref):
        if False:
            print('Hello World!')
        '\n        All rules that reference the supplied trigger_ref.\n        '
        return Rule.get_all(**{'trigger': trigger_ref})

    def _update_trigger_ref_count(self, trigger_db, ref_count):
        if False:
            print('Hello World!')
        '\n        Non-publishing ref_count update to a TriggerDB.\n        '
        trigger_db.ref_count = ref_count
        Trigger.add_or_update(trigger_db, publish=False, dispatch_trigger=False)

    def migrate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Will migrate all Triggers that should have ref_count to have the right ref_count.\n        '
        trigger_dbs = self._get_trigger_with_parameters()
        for trigger_db in trigger_dbs:
            trigger_ref = trigger_db.get_reference().ref
            rules = self._get_rules_for_trigger(trigger_ref=trigger_ref)
            ref_count = len(rules)
            print('Updating Trigger %s to ref_count %s' % (trigger_ref, ref_count))
            self._update_trigger_ref_count(trigger_db=trigger_db, ref_count=ref_count)

def setup():
    if False:
        i = 10
        return i + 15
    common_setup(config=config, setup_db=True, register_mq_exchanges=True)

def teartown():
    if False:
        print('Hello World!')
    common_teardown()

def main():
    if False:
        while True:
            i = 10
    setup()
    try:
        TriggerMigrator().migrate()
    finally:
        teartown()
if __name__ == '__main__':
    main()