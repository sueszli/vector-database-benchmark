from __future__ import absolute_import
from kombu import Exchange, Queue
from st2common.transport import publishers
__all__ = ['ActionExecutionStatePublisher']
ACTIONEXECUTIONSTATE_XCHG = Exchange('st2.actionexecutionstate', type='topic')

class ActionExecutionStatePublisher(publishers.CUDPublisher):

    def __init__(self):
        if False:
            return 10
        super(ActionExecutionStatePublisher, self).__init__(exchange=ACTIONEXECUTIONSTATE_XCHG)

def get_queue(name, routing_key):
    if False:
        for i in range(10):
            print('nop')
    return Queue(name, ACTIONEXECUTIONSTATE_XCHG, routing_key=routing_key)