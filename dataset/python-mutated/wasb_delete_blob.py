from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence
from airflow.models import BaseOperator
from airflow.providers.microsoft.azure.hooks.wasb import WasbHook
if TYPE_CHECKING:
    from airflow.utils.context import Context

class WasbDeleteBlobOperator(BaseOperator):
    """
    Deletes blob(s) on Azure Blob Storage.

    :param container_name: Name of the container. (templated)
    :param blob_name: Name of the blob. (templated)
    :param wasb_conn_id: Reference to the :ref:`wasb connection <howto/connection:wasb>`.
    :param check_options: Optional keyword arguments that
        `WasbHook.check_for_blob()` takes.
    :param is_prefix: If blob_name is a prefix, delete all files matching prefix.
    :param ignore_if_missing: if True, then return success even if the
        blob does not exist.
    """
    template_fields: Sequence[str] = ('container_name', 'blob_name')

    def __init__(self, *, container_name: str, blob_name: str, wasb_conn_id: str='wasb_default', check_options: Any=None, is_prefix: bool=False, ignore_if_missing: bool=False, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        if check_options is None:
            check_options = {}
        self.wasb_conn_id = wasb_conn_id
        self.container_name = container_name
        self.blob_name = blob_name
        self.check_options = check_options
        self.is_prefix = is_prefix
        self.ignore_if_missing = ignore_if_missing

    def execute(self, context: Context) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.log.info('Deleting blob: %s\n in wasb://%s', self.blob_name, self.container_name)
        hook = WasbHook(wasb_conn_id=self.wasb_conn_id)
        hook.delete_file(self.container_name, self.blob_name, self.is_prefix, self.ignore_if_missing, **self.check_options)