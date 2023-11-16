from __future__ import annotations
import logging
from ipv8.community import Community
from ipv8.peerdiscovery.discovery import DiscoveryStrategy, EdgeWalk

class DiscoveryBooster:
    """This class is designed for increasing the speed of peers' discovery during a limited time.

    It can be applied to any community.
    """

    def __init__(self, timeout_in_sec: float=120.0, take_step_interval_in_sec: float=1.1, walker: DiscoveryStrategy=None):
        if False:
            print('Hello World!')
        "\n\n        Args:\n            timeout_in_sec: DiscoveryBooster work timeout. When this timeout will be reached,\n                `finish` function will be called.\n            take_step_interval_in_sec: Сall frequency of walker's `take_step` function.\n            walker: walker that will be used during boost period.\n        "
        self.logger = logging.getLogger(self.__class__.__name__)
        self.timeout_in_sec = timeout_in_sec
        self.take_step_interval_in_sec = take_step_interval_in_sec
        self.walker = walker
        self.community = None
        self._take_step_task_name = 'take step'

    def apply(self, community: Community):
        if False:
            for i in range(10):
                print('nop')
        'Apply DiscoveryBooster to the community\n\n        Args:\n            community: community to implement DiscoveryBooster\n\n        Returns: None\n        '
        if not community:
            return
        self.logger.info(f'Apply. Timeout: {self.timeout_in_sec}s. Take step interval: {self.take_step_interval_in_sec}s')
        self.community = community
        if not self.walker:
            self.walker = EdgeWalk(community, neighborhood_size=25, edge_length=25)
        community.register_task(self._take_step_task_name, self.take_step, interval=self.take_step_interval_in_sec)
        community.register_task('finish', self.finish, delay=self.timeout_in_sec)

    def finish(self):
        if False:
            return 10
        "Finish DiscoveryBooster work.\n\n        This function returns defaults max_peers to the community.\n\n        Will be called automatically from community's task manager.\n\n        Returns: None\n        "
        self.logger.info(f'Finish. Cancel pending task: {self._take_step_task_name}')
        self.community.cancel_pending_task(self._take_step_task_name)

    def take_step(self):
        if False:
            i = 10
            return i + 15
        "Take a step by invoke `walker.take_step()`\n\n        Will be called automatically from community's task manager.\n\n        Returns: None\n        "
        self.logger.debug('Take a step')
        self.walker.take_step()