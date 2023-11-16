from abc import ABCMeta
from datetime import timedelta
from typing import Optional
from py4j.java_gateway import get_java_class
from pyflink.java_gateway import get_gateway
from pyflink.util.java_utils import to_j_flink_time, from_j_flink_time
__all__ = ['RestartStrategies', 'RestartStrategyConfiguration']

class RestartStrategyConfiguration(object, metaclass=ABCMeta):
    """
    Abstract configuration for restart strategies.
    """

    def __init__(self, j_restart_strategy_configuration):
        if False:
            i = 10
            return i + 15
        self._j_restart_strategy_configuration = j_restart_strategy_configuration

    def get_description(self) -> str:
        if False:
            return 10
        '\n        Returns a description which is shown in the web interface.\n\n        :return: Description of the restart strategy.\n        '
        return self._j_restart_strategy_configuration.getDescription()

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, self.__class__) and self._j_restart_strategy_configuration == other._j_restart_strategy_configuration

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._j_restart_strategy_configuration.hashCode()

class RestartStrategies(object):
    """
    This class defines methods to generate RestartStrategyConfigurations. These configurations are
    used to create RestartStrategies at runtime.

    The RestartStrategyConfigurations are used to decouple the core module from the runtime module.
    """

    class NoRestartStrategyConfiguration(RestartStrategyConfiguration):
        """
        Configuration representing no restart strategy.
        """

        def __init__(self, j_restart_strategy=None):
            if False:
                print('Hello World!')
            if j_restart_strategy is None:
                gateway = get_gateway()
                self._j_restart_strategy_configuration = gateway.jvm.RestartStrategies.NoRestartStrategyConfiguration()
                super(RestartStrategies.NoRestartStrategyConfiguration, self).__init__(self._j_restart_strategy_configuration)
            else:
                super(RestartStrategies.NoRestartStrategyConfiguration, self).__init__(j_restart_strategy)

    class FixedDelayRestartStrategyConfiguration(RestartStrategyConfiguration):
        """
        Configuration representing a fixed delay restart strategy.
        """

        def __init__(self, restart_attempts=None, delay_between_attempts_interval=None, j_restart_strategy=None):
            if False:
                i = 10
                return i + 15
            if j_restart_strategy is None:
                if not isinstance(delay_between_attempts_interval, (timedelta, int)):
                    raise TypeError("The delay_between_attempts_interval 'failure_interval' only supports integer and datetime.timedelta, current input type is %s." % type(delay_between_attempts_interval))
                gateway = get_gateway()
                self._j_restart_strategy_configuration = gateway.jvm.RestartStrategies.fixedDelayRestart(restart_attempts, to_j_flink_time(delay_between_attempts_interval))
                super(RestartStrategies.FixedDelayRestartStrategyConfiguration, self).__init__(self._j_restart_strategy_configuration)
            else:
                super(RestartStrategies.FixedDelayRestartStrategyConfiguration, self).__init__(j_restart_strategy)

        def get_restart_attempts(self) -> int:
            if False:
                return 10
            return self._j_restart_strategy_configuration.getRestartAttempts()

        def get_delay_between_attempts_interval(self) -> timedelta:
            if False:
                return 10
            return from_j_flink_time(self._j_restart_strategy_configuration.getDelayBetweenAttemptsInterval())

    class FailureRateRestartStrategyConfiguration(RestartStrategyConfiguration):
        """
        Configuration representing a failure rate restart strategy.
        """

        def __init__(self, max_failure_rate=None, failure_interval=None, delay_between_attempts_interval=None, j_restart_strategy=None):
            if False:
                print('Hello World!')
            if j_restart_strategy is None:
                if not isinstance(failure_interval, (timedelta, int)):
                    raise TypeError("The parameter 'failure_interval' only supports integer and datetime.timedelta, current input type is %s." % type(failure_interval))
                if not isinstance(delay_between_attempts_interval, (timedelta, int)):
                    raise TypeError("The delay_between_attempts_interval 'failure_interval' only supports integer and datetime.timedelta, current input type is %s." % type(delay_between_attempts_interval))
                gateway = get_gateway()
                self._j_restart_strategy_configuration = gateway.jvm.RestartStrategies.FailureRateRestartStrategyConfiguration(max_failure_rate, to_j_flink_time(failure_interval), to_j_flink_time(delay_between_attempts_interval))
                super(RestartStrategies.FailureRateRestartStrategyConfiguration, self).__init__(self._j_restart_strategy_configuration)
            else:
                super(RestartStrategies.FailureRateRestartStrategyConfiguration, self).__init__(j_restart_strategy)

        def get_max_failure_rate(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return self._j_restart_strategy_configuration.getMaxFailureRate()

        def get_failure_interval(self) -> timedelta:
            if False:
                return 10
            return from_j_flink_time(self._j_restart_strategy_configuration.getFailureInterval())

        def get_delay_between_attempts_interval(self) -> timedelta:
            if False:
                return 10
            return from_j_flink_time(self._j_restart_strategy_configuration.getDelayBetweenAttemptsInterval())

    class FallbackRestartStrategyConfiguration(RestartStrategyConfiguration):
        """
        Restart strategy configuration that could be used by jobs to use cluster level restart
        strategy. Useful especially when one has a custom implementation of restart strategy set via
        flink-conf.yaml.
        """

        def __init__(self, j_restart_strategy=None):
            if False:
                while True:
                    i = 10
            if j_restart_strategy is None:
                gateway = get_gateway()
                self._j_restart_strategy_configuration = gateway.jvm.RestartStrategies.FallbackRestartStrategyConfiguration()
                super(RestartStrategies.FallbackRestartStrategyConfiguration, self).__init__(self._j_restart_strategy_configuration)
            else:
                super(RestartStrategies.FallbackRestartStrategyConfiguration, self).__init__(j_restart_strategy)

    @staticmethod
    def _from_j_restart_strategy(j_restart_strategy) -> Optional[RestartStrategyConfiguration]:
        if False:
            i = 10
            return i + 15
        if j_restart_strategy is None:
            return None
        gateway = get_gateway()
        NoRestartStrategyConfiguration = gateway.jvm.RestartStrategies.NoRestartStrategyConfiguration
        FixedDelayRestartStrategyConfiguration = gateway.jvm.RestartStrategies.FixedDelayRestartStrategyConfiguration
        FailureRateRestartStrategyConfiguration = gateway.jvm.RestartStrategies.FailureRateRestartStrategyConfiguration
        FallbackRestartStrategyConfiguration = gateway.jvm.RestartStrategies.FallbackRestartStrategyConfiguration
        clz = j_restart_strategy.getClass()
        if clz.getName() == get_java_class(NoRestartStrategyConfiguration).getName():
            return RestartStrategies.NoRestartStrategyConfiguration(j_restart_strategy=j_restart_strategy)
        elif clz.getName() == get_java_class(FixedDelayRestartStrategyConfiguration).getName():
            return RestartStrategies.FixedDelayRestartStrategyConfiguration(j_restart_strategy=j_restart_strategy)
        elif clz.getName() == get_java_class(FailureRateRestartStrategyConfiguration).getName():
            return RestartStrategies.FailureRateRestartStrategyConfiguration(j_restart_strategy=j_restart_strategy)
        elif clz.getName() == get_java_class(FallbackRestartStrategyConfiguration).getName():
            return RestartStrategies.FallbackRestartStrategyConfiguration(j_restart_strategy=j_restart_strategy)
        else:
            raise Exception('Unsupported java RestartStrategyConfiguration: %s' % clz.getName())

    @staticmethod
    def no_restart() -> 'NoRestartStrategyConfiguration':
        if False:
            print('Hello World!')
        '\n        Generates NoRestartStrategyConfiguration.\n\n        :return: The :class:`NoRestartStrategyConfiguration`.\n        '
        return RestartStrategies.NoRestartStrategyConfiguration()

    @staticmethod
    def fall_back_restart() -> 'FallbackRestartStrategyConfiguration':
        if False:
            print('Hello World!')
        return RestartStrategies.FallbackRestartStrategyConfiguration()

    @staticmethod
    def fixed_delay_restart(restart_attempts: int, delay_between_attempts: int) -> 'FixedDelayRestartStrategyConfiguration':
        if False:
            while True:
                i = 10
        '\n        Generates a FixedDelayRestartStrategyConfiguration.\n\n        :param restart_attempts: Number of restart attempts for the FixedDelayRestartStrategy.\n        :param delay_between_attempts: Delay in-between restart attempts for the\n                                       FixedDelayRestartStrategy, the input could be integer value\n                                       in milliseconds or datetime.timedelta object.\n        :return: The :class:`FixedDelayRestartStrategyConfiguration`.\n        '
        return RestartStrategies.FixedDelayRestartStrategyConfiguration(restart_attempts, delay_between_attempts)

    @staticmethod
    def failure_rate_restart(failure_rate: int, failure_interval: int, delay_interval: int) -> 'FailureRateRestartStrategyConfiguration':
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates a FailureRateRestartStrategyConfiguration.\n\n        :param failure_rate: Maximum number of restarts in given interval ``failure_interval``\n                             before failing a job.\n        :param failure_interval: Time interval for failures, the input could be integer value\n                                 in milliseconds or datetime.timedelta object.\n        :param delay_interval: Delay in-between restart attempts, the input could be integer value\n                               in milliseconds or datetime.timedelta object.\n        '
        return RestartStrategies.FailureRateRestartStrategyConfiguration(failure_rate, failure_interval, delay_interval)