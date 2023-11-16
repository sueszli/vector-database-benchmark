import logging
import statistics
import time
from ipv8.REST.asyncio_endpoint import DriftMeasurementStrategy
from ipv8.taskmanager import TaskManager
from ipv8_service import IPv8

class IPv8Monitor:
    """
    Monitor the state of IPv8 and adjust its walk_interval accordingly.
    """

    def __init__(self, ipv8_instance: IPv8, min_update_rate: float, max_update_rate: float, choke_limit: float=0.01) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Monitor an IPv8 instance and modulate its walk_interval between the minimum and maximum update rates.\n        This uses a hard limit to distinguish choke from noise in the congestion.\n\n        :param ipv8_instance: the IPv8 instance to modulate the walk_interval for.\n        :param min_update_rate: the minimum time between steps (in seconds).\n        :param max_update_rate: the maximum time between steps (in seconds).\n        :param choke_limit: the noise limit for choke detection (in seconds).\n        '
        super().__init__()
        self.ipv8_instance = ipv8_instance
        self.min_update_rate = min_update_rate
        self.max_update_rate = max_update_rate
        self.choke_limit = choke_limit
        self.measurement_strategy = DriftMeasurementStrategy(min_update_rate)
        self.current_rate = min_update_rate
        self.last_check = time.time()
        self.interval = 5.0
        self.speedup_step = (self.max_update_rate - self.min_update_rate) / 5.0
        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self, task_manager: TaskManager, interval: float=5.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Insert this monitor into the IPv8 strategies and start scaling periodically scaling the walk_interval.\n\n        :param task_manager: The TaskManager to register our periodic check for.\n        :param interval: The time (in seconds) between checking the IPv8 health.\n        '
        self.interval = interval
        with self.ipv8_instance.overlay_lock:
            self.ipv8_instance.strategies.append((self.measurement_strategy, -1))
        task_manager.register_task('IPv8 auto-scaling', self.auto_scale_ipv8, interval=interval)

    def auto_scale_ipv8(self):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate the IPv8 choking history and determine whether the walk_interval should be slowed down or sped up.\n        '
        if self.last_check + self.interval > time.time():
            return
        self.last_check = time.time()
        history = [record[1] for record in self.measurement_strategy.history if record[0] > self.last_check - self.interval]
        median_time_taken = statistics.median(history) if history else 0.0
        self.logger.debug('Mean drift: %f, choke detected: %s.', median_time_taken, str(median_time_taken > self.choke_limit))
        if median_time_taken > self.choke_limit:
            self.current_rate = min(self.max_update_rate, self.current_rate * 1.5)
        else:
            self.current_rate = max(self.min_update_rate, self.current_rate - self.speedup_step)
        self.logger.debug('Current walk_interval: %f.', self.current_rate)
        self.ipv8_instance.walk_interval = self.current_rate
        with self.ipv8_instance.overlay_lock:
            for (strategy, _) in self.ipv8_instance.strategies:
                if isinstance(strategy, DriftMeasurementStrategy):
                    strategy.core_update_rate = self.current_rate