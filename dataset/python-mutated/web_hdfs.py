from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence
from airflow.sensors.base import BaseSensorOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class WebHdfsSensor(BaseSensorOperator):
    """Waits for a file or folder to land in HDFS."""
    template_fields: Sequence[str] = ('filepath',)

    def __init__(self, *, filepath: str, webhdfs_conn_id: str='webhdfs_default', **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.filepath = filepath
        self.webhdfs_conn_id = webhdfs_conn_id

    def poke(self, context: Context) -> bool:
        if False:
            print('Hello World!')
        from airflow.providers.apache.hdfs.hooks.webhdfs import WebHDFSHook
        hook = WebHDFSHook(self.webhdfs_conn_id)
        self.log.info('Poking for file %s', self.filepath)
        return hook.check_for_path(hdfs_path=self.filepath)