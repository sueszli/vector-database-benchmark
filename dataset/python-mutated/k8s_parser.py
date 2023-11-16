import os
import numpy as np
from time import sleep
from typing import Dict, List, Optional

class K8SParser:

    def __init__(self, platform_spec: Optional[Dict]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Should only set global cluster properties\n        '
        self.kwargs = kwargs
        self.nodelist = self._parse_node_list()
        self.ntasks = len(self.nodelist)
        self.platform_spec = platform_spec
        self.parallel_workers = kwargs.get('parallel_workers') or 1
        self.topology = kwargs.get('topology') or 'alone'
        self.ports = int(kwargs.get('ports') or 50515)
        self.tasks = {}

    def parse(self) -> dict:
        if False:
            return 10
        if self.kwargs.get('mq_type', 'nng') != 'nng':
            return self.kwargs
        procid = int(os.environ['DI_RANK'])
        nodename = self.nodelist[procid]
        task = self._get_task(procid)
        assert task['address'] == nodename
        return {**self.kwargs, **task}

    def _parse_node_list(self) -> List[str]:
        if False:
            return 10
        return os.environ['DI_NODES'].split(',')

    def _get_task(self, procid: int) -> dict:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Complete node properties, use environment vars in list instead of on current node.\n            For example, if you want to set nodename in this function, please derive it from DI_NODES.\n        Arguments:\n            - procid (:obj:`int`): Proc order, starting from 0, must be set automatically by dijob.\n                Note that it is different from node_id.\n        '
        if procid in self.tasks:
            return self.tasks.get(procid)
        if self.platform_spec:
            task = self.platform_spec['tasks'][procid]
        else:
            task = {}
        if 'ports' not in task:
            task['ports'] = self.kwargs.get('ports') or self._get_ports()
        if 'address' not in task:
            task['address'] = self.kwargs.get('address') or self._get_address(procid)
        if 'node_ids' not in task:
            task['node_ids'] = self.kwargs.get('node_ids') or self._get_node_id(procid)
        task['attach_to'] = self.kwargs.get('attach_to') or self._get_attach_to(procid, task.get('attach_to'))
        task['topology'] = self.topology
        task['parallel_workers'] = self.parallel_workers
        self.tasks[procid] = task
        return task

    def _get_attach_to(self, procid: int, attach_to: Optional[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Parse from pattern of attach_to. If attach_to is specified in the platform_spec,\n            it is formatted as a real address based on the specified address.\n            If not, the real addresses will be generated based on the globally specified typology.\n        Arguments:\n            - procid (:obj:`int`): Proc order.\n            - attach_to (:obj:`str`): The attach_to field in platform_spec for the task with current procid.\n        Returns\n            - attach_to (:obj:`str`): The real addresses for attach_to.\n        '
        if attach_to:
            attach_to = [self._get_attach_to_part(part) for part in attach_to.split(',')]
        elif procid == 0:
            attach_to = []
        elif self.topology == 'mesh':
            prev_tasks = [self._get_task(i) for i in range(procid)]
            attach_to = [self._get_attach_to_from_task(task) for task in prev_tasks]
            attach_to = list(np.concatenate(attach_to))
        elif self.topology == 'star':
            head_task = self._get_task(0)
            attach_to = self._get_attach_to_from_task(head_task)
        else:
            attach_to = []
        return ','.join(attach_to)

    def _get_attach_to_part(self, attach_part: str) -> str:
        if False:
            return 10
        '\n        Overview:\n            Parse each part of attach_to.\n        Arguments:\n            - attach_part (:obj:`str`): The attach_to field with specific pattern, e.g. $node:0\n        Returns\n            - attach_to (:obj:`str`): The real address, e.g. tcp://SH-0:50000\n        '
        if not attach_part.startswith('$node.'):
            return attach_part
        attach_node_id = int(attach_part[6:])
        attach_task = self._get_task(self._get_procid_from_nodeid(attach_node_id))
        return self._get_tcp_link(attach_task['address'], attach_task['ports'])

    def _get_attach_to_from_task(self, task: dict) -> List[str]:
        if False:
            return 10
        '\n        Overview:\n            Get attach_to list from task, note that parallel_workers will affact the connected processes.\n        Arguments:\n            - task (:obj:`dict`): The task object.\n        Returns\n            - attach_to (:obj:`str`): The real address, e.g. tcp://SH-0:50000\n        '
        port = task.get('ports')
        address = task.get('address')
        ports = [int(port) + i for i in range(self.parallel_workers)]
        attach_to = [self._get_tcp_link(address, port) for port in ports]
        return attach_to

    def _get_procid_from_nodeid(self, nodeid: int) -> int:
        if False:
            while True:
                i = 10
        procid = None
        for i in range(self.ntasks):
            task = self._get_task(i)
            if task['node_ids'] == nodeid:
                procid = i
                break
        if procid is None:
            raise Exception('Can not find procid from nodeid: {}'.format(nodeid))
        return procid

    def _get_ports(self) -> str:
        if False:
            while True:
                i = 10
        return self.ports

    def _get_address(self, procid: int) -> str:
        if False:
            return 10
        address = self.nodelist[procid]
        return address

    def _get_tcp_link(self, address: str, port: int) -> str:
        if False:
            return 10
        return 'tcp://{}:{}'.format(address, port)

    def _get_node_id(self, procid: int) -> int:
        if False:
            i = 10
            return i + 15
        return procid * self.parallel_workers

def k8s_parser(platform_spec: Optional[str]=None, **kwargs) -> dict:
    if False:
        return 10
    return K8SParser(platform_spec, **kwargs).parse()