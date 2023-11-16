from __future__ import annotations
import logging
from functools import wraps
from typing import TYPE_CHECKING, Callable, TypeVar, cast
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException
from airflow.metrics.protocols import Timer
from airflow.metrics.validators import AllowListValidator, BlockListValidator, validate_stat
if TYPE_CHECKING:
    from statsd import StatsClient
    from airflow.metrics.protocols import DeltaType, TimerProtocol
    from airflow.metrics.validators import ListValidator
T = TypeVar('T', bound=Callable)
log = logging.getLogger(__name__)

def prepare_stat_with_tags(fn: T) -> T:
    if False:
        for i in range(10):
            print('nop')
    'Add tags to stat with influxdb standard format if influxdb_tags_enabled is True.'

    @wraps(fn)
    def wrapper(self, stat: str | None=None, *args, tags: dict[str, str] | None=None, **kwargs) -> Callable[[str], str]:
        if False:
            i = 10
            return i + 15
        if self.influxdb_tags_enabled:
            if stat is not None and tags is not None:
                for (k, v) in tags.items():
                    if self.metric_tags_validator.test(k):
                        if all((c not in [',', '='] for c in v + k)):
                            stat += f',{k}={v}'
                        else:
                            log.error('Dropping invalid tag: %s=%s.', k, v)
        return fn(self, stat, *args, tags=tags, **kwargs)
    return cast(T, wrapper)

class SafeStatsdLogger:
    """StatsD Logger."""

    def __init__(self, statsd_client: StatsClient, metrics_validator: ListValidator=AllowListValidator(), influxdb_tags_enabled: bool=False, metric_tags_validator: ListValidator=AllowListValidator()) -> None:
        if False:
            print('Hello World!')
        self.statsd = statsd_client
        self.metrics_validator = metrics_validator
        self.influxdb_tags_enabled = influxdb_tags_enabled
        self.metric_tags_validator = metric_tags_validator

    @prepare_stat_with_tags
    @validate_stat
    def incr(self, stat: str, count: int=1, rate: float=1, *, tags: dict[str, str] | None=None) -> None:
        if False:
            return 10
        'Increment stat.'
        if self.metrics_validator.test(stat):
            return self.statsd.incr(stat, count, rate)
        return None

    @prepare_stat_with_tags
    @validate_stat
    def decr(self, stat: str, count: int=1, rate: float=1, *, tags: dict[str, str] | None=None) -> None:
        if False:
            while True:
                i = 10
        'Decrement stat.'
        if self.metrics_validator.test(stat):
            return self.statsd.decr(stat, count, rate)
        return None

    @prepare_stat_with_tags
    @validate_stat
    def gauge(self, stat: str, value: int | float, rate: float=1, delta: bool=False, *, tags: dict[str, str] | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'Gauge stat.'
        if self.metrics_validator.test(stat):
            return self.statsd.gauge(stat, value, rate, delta)
        return None

    @prepare_stat_with_tags
    @validate_stat
    def timing(self, stat: str, dt: DeltaType, *, tags: dict[str, str] | None=None) -> None:
        if False:
            while True:
                i = 10
        'Stats timing.'
        if self.metrics_validator.test(stat):
            return self.statsd.timing(stat, dt)
        return None

    @prepare_stat_with_tags
    @validate_stat
    def timer(self, stat: str | None=None, *args, tags: dict[str, str] | None=None, **kwargs) -> TimerProtocol:
        if False:
            print('Hello World!')
        'Timer metric that can be cancelled.'
        if stat and self.metrics_validator.test(stat):
            return Timer(self.statsd.timer(stat, *args, **kwargs))
        return Timer()

def get_statsd_logger(cls) -> SafeStatsdLogger:
    if False:
        while True:
            i = 10
    'Return logger for StatsD.'
    from statsd import StatsClient
    stats_class = conf.getimport('metrics', 'statsd_custom_client_path', fallback=None)
    metrics_validator: ListValidator
    if stats_class:
        if not issubclass(stats_class, StatsClient):
            raise AirflowConfigException('Your custom StatsD client must extend the statsd.StatsClient in order to ensure backwards compatibility.')
        else:
            log.info('Successfully loaded custom StatsD client')
    else:
        stats_class = StatsClient
    statsd = stats_class(host=conf.get('metrics', 'statsd_host'), port=conf.getint('metrics', 'statsd_port'), prefix=conf.get('metrics', 'statsd_prefix'))
    if conf.get('metrics', 'metrics_allow_list', fallback=None):
        metrics_validator = AllowListValidator(conf.get('metrics', 'metrics_allow_list'))
        if conf.get('metrics', 'metrics_block_list', fallback=None):
            log.warning('Ignoring metrics_block_list as both metrics_allow_list and metrics_block_list have been set')
    elif conf.get('metrics', 'metrics_block_list', fallback=None):
        metrics_validator = BlockListValidator(conf.get('metrics', 'metrics_block_list'))
    else:
        metrics_validator = AllowListValidator()
    influxdb_tags_enabled = conf.getboolean('metrics', 'statsd_influxdb_enabled', fallback=False)
    metric_tags_validator = BlockListValidator(conf.get('metrics', 'statsd_disabled_tags', fallback=None))
    return SafeStatsdLogger(statsd, metrics_validator, influxdb_tags_enabled, metric_tags_validator)