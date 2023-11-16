from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Any, Sequence
from deprecated.classic import deprecated
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.alibaba.cloud.hooks.analyticdb_spark import AnalyticDBSparkHook, AppState
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class AnalyticDBSparkSensor(BaseSensorOperator):
    """
    Monitor a AnalyticDB Spark session for termination.

    :param app_id: identifier of the monitored app depends on the option that's being modified.
    :param adb_spark_conn_id: reference to a pre-defined ADB Spark connection.
    :param region: AnalyticDB MySQL region you want to submit spark application.
    """
    template_fields: Sequence[str] = ('app_id',)

    def __init__(self, *, app_id: str, adb_spark_conn_id: str='adb_spark_default', region: str | None=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.app_id = app_id
        self._region = region
        self._adb_spark_conn_id = adb_spark_conn_id

    @cached_property
    def hook(self) -> AnalyticDBSparkHook:
        if False:
            print('Hello World!')
        'Get valid hook.'
        return AnalyticDBSparkHook(adb_spark_conn_id=self._adb_spark_conn_id, region=self._region)

    @deprecated(reason='use `hook` property instead.', category=AirflowProviderDeprecationWarning)
    def get_hook(self) -> AnalyticDBSparkHook:
        if False:
            i = 10
            return i + 15
        'Get valid hook.'
        return self.hook

    def poke(self, context: Context) -> bool:
        if False:
            while True:
                i = 10
        app_id = self.app_id
        state = self.hook.get_spark_state(app_id)
        return AppState(state) in AnalyticDBSparkHook.TERMINAL_STATES