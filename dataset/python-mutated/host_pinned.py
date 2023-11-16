from __future__ import annotations
DOCUMENTATION = '\n    name: host_pinned\n    short_description: Executes tasks on each host without interruption\n    description:\n        - Task execution is as fast as possible per host in batch as defined by C(serial) (default all).\n          Ansible will not start a play for a host unless the play can be finished without interruption by tasks for another host,\n          i.e. the number of hosts with an active play does not exceed the number of forks.\n          Ansible will not wait for other hosts to finish the current task before queuing the next task for a host that has finished.\n          Once a host is done with the play, it opens it\'s slot to a new host that was waiting to start.\n          Other than that, it behaves just like the "free" strategy.\n    version_added: "2.7"\n    author: Ansible Core Team\n'
from ansible.plugins.strategy.free import StrategyModule as FreeStrategyModule
from ansible.utils.display import Display
display = Display()

class StrategyModule(FreeStrategyModule):

    def __init__(self, tqm):
        if False:
            for i in range(10):
                print('nop')
        super(StrategyModule, self).__init__(tqm)
        self._host_pinned = True