import logging
import math
from abc import ABCMeta, abstractmethod
from typing import List
from ray.serve._private.constants import CONTROL_LOOP_PERIOD_S, SERVE_LOGGER_NAME
from ray.serve.config import AutoscalingConfig
logger = logging.getLogger(SERVE_LOGGER_NAME)

def calculate_desired_num_replicas(autoscaling_config: AutoscalingConfig, current_num_ongoing_requests: List[float]) -> int:
    if False:
        print('Hello World!')
    'Returns the number of replicas to scale to based on the given metrics.\n\n    Args:\n        autoscaling_config: The autoscaling parameters to use for this\n            calculation.\n        current_num_ongoing_requests (List[float]): A list of the number of\n            ongoing requests for each replica.  Assumes each entry has already\n            been time-averaged over the desired lookback window.\n\n    Returns:\n        desired_num_replicas: The desired number of replicas to scale to, based\n            on the input metrics and the current number of replicas.\n\n    '
    current_num_replicas = len(current_num_ongoing_requests)
    if current_num_replicas == 0:
        raise ValueError('Number of replicas cannot be zero')
    num_ongoing_requests_per_replica: float = sum(current_num_ongoing_requests) / len(current_num_ongoing_requests)
    error_ratio: float = num_ongoing_requests_per_replica / autoscaling_config.target_num_ongoing_requests_per_replica
    if error_ratio >= 1:
        smoothing_factor = autoscaling_config.get_upscale_smoothing_factor()
    else:
        smoothing_factor = autoscaling_config.get_downscale_smoothing_factor()
    smoothed_error_ratio = 1 + (error_ratio - 1) * smoothing_factor
    desired_num_replicas = math.ceil(current_num_replicas * smoothed_error_ratio)
    if error_ratio == 0 and desired_num_replicas == current_num_replicas and (desired_num_replicas >= 1):
        desired_num_replicas -= 1
    desired_num_replicas = min(autoscaling_config.max_replicas, desired_num_replicas)
    desired_num_replicas = max(autoscaling_config.min_replicas, desired_num_replicas)
    return desired_num_replicas

class AutoscalingPolicy:
    """Defines the interface for an autoscaling policy.

    To add a new autoscaling policy, a class should be defined that provides
    this interface. The class may be stateful, in which case it may also want
    to provide a non-default constructor. However, this state will be lost when
    the controller recovers from a failure.
    """
    __metaclass__ = ABCMeta

    def __init__(self, config: AutoscalingConfig):
        if False:
            return 10
        'Initialize the policy using the specified config dictionary.'
        self.config = config

    @abstractmethod
    def get_decision_num_replicas(self, curr_target_num_replicas: int, current_num_ongoing_requests: List[float], current_handle_queued_queries: float) -> int:
        if False:
            i = 10
            return i + 15
        'Make a decision to scale replicas.\n\n        Arguments:\n            current_num_ongoing_requests: List[float]: List of number of\n                ongoing requests for each replica.\n            curr_target_num_replicas: The number of replicas that the\n                deployment is currently trying to scale to.\n            current_handle_queued_queries : The number of handle queued queries,\n                if there are multiple handles, the max number of queries at\n                a single handle should be passed in\n\n        Returns:\n            int: The new number of replicas to scale to.\n        '
        return curr_target_num_replicas

class BasicAutoscalingPolicy(AutoscalingPolicy):
    """The default autoscaling policy based on basic thresholds for scaling.
    There is a minimum threshold for the average queue length in the cluster
    to scale up and a maximum threshold to scale down. Each period, a 'scale
    up' or 'scale down' decision is made. This decision must be made for a
    specified number of periods in a row before the number of replicas is
    actually scaled. See config options for more details.  Assumes
    `get_decision_num_replicas` is called once every CONTROL_LOOP_PERIOD_S
    seconds.
    """

    def __init__(self, config: AutoscalingConfig):
        if False:
            print('Hello World!')
        self.config = config
        self.loop_period_s = CONTROL_LOOP_PERIOD_S
        self.scale_up_consecutive_periods = int(config.upscale_delay_s / self.loop_period_s)
        self.scale_down_consecutive_periods = int(config.downscale_delay_s / self.loop_period_s)
        self.decision_counter = 0

    def get_decision_num_replicas(self, curr_target_num_replicas: int, current_num_ongoing_requests: List[float], current_handle_queued_queries: float) -> int:
        if False:
            i = 10
            return i + 15
        if len(current_num_ongoing_requests) == 0:
            if current_handle_queued_queries > 0:
                return max(math.ceil(1 * self.config.get_upscale_smoothing_factor()), curr_target_num_replicas)
            return curr_target_num_replicas
        decision_num_replicas = curr_target_num_replicas
        desired_num_replicas = calculate_desired_num_replicas(self.config, current_num_ongoing_requests)
        if desired_num_replicas > curr_target_num_replicas:
            if self.decision_counter < 0:
                self.decision_counter = 0
            self.decision_counter += 1
            if self.decision_counter > self.scale_up_consecutive_periods:
                self.decision_counter = 0
                decision_num_replicas = desired_num_replicas
        elif desired_num_replicas < curr_target_num_replicas:
            if self.decision_counter > 0:
                self.decision_counter = 0
            self.decision_counter -= 1
            if self.decision_counter < -self.scale_down_consecutive_periods:
                self.decision_counter = 0
                decision_num_replicas = desired_num_replicas
        else:
            self.decision_counter = 0
        return decision_num_replicas