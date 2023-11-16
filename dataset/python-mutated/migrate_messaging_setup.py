"""
A migration script that cleans up old queues.
"""
from __future__ import absolute_import
import traceback
from st2common import config
from st2common.transport import reactor
from st2common.transport import utils as transport_utils

class Migrate_0_13_x_to_1_1_0(object):
    """
    Handles migration of messaging setup from 0.13.x to 1.1.
    """
    OLD_QS = [reactor.get_trigger_cud_queue('st2.trigger.watch.timers', routing_key='#'), reactor.get_trigger_cud_queue('st2.trigger.watch.sensorwrapper', routing_key='#'), reactor.get_trigger_cud_queue('st2.trigger.watch.webhooks', routing_key='#')]

    def migrate(self):
        if False:
            i = 10
            return i + 15
        self._cleanup_old_queues()

    def _cleanup_old_queues(self):
        if False:
            return 10
        with transport_utils.get_connection() as connection:
            for q in self.OLD_QS:
                bound_q = q(connection.default_channel)
                try:
                    bound_q.delete()
                except:
                    print('Failed to delete %s.' % q.name)
                    traceback.print_exc()

def main():
    if False:
        print('Hello World!')
    try:
        migrator = Migrate_0_13_x_to_1_1_0()
        migrator.migrate()
    except:
        print('Messaging setup migration failed.')
        traceback.print_exc()
if __name__ == '__main__':
    config.parse_args(args={})
    main()