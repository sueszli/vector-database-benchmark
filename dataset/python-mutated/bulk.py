from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.models import BaseOperator
from airflow.providers.salesforce.hooks.salesforce import SalesforceHook
if TYPE_CHECKING:
    from typing_extensions import Literal
    from airflow.utils.context import Context

class SalesforceBulkOperator(BaseOperator):
    """
    Execute a Salesforce Bulk API and pushes results to xcom.

    .. seealso::
        For more information on how to use this operator, take a look at the guide:
        :ref:`howto/operator:SalesforceBulkOperator`

    :param operation: Bulk operation to be performed
        Available operations are in ['insert', 'update', 'upsert', 'delete', 'hard_delete']
    :param object_name: The name of the Salesforce object
    :param payload: list of dict to be passed as a batch
    :param external_id_field: unique identifier field for upsert operations
    :param batch_size: number of records to assign for each batch in the job
    :param use_serial: Process batches in serial mode
    :param salesforce_conn_id: The :ref:`Salesforce Connection id <howto/connection:SalesforceHook>`.
    """
    available_operations = ('insert', 'update', 'upsert', 'delete', 'hard_delete')

    def __init__(self, *, operation: Literal['insert', 'update', 'upsert', 'delete', 'hard_delete'], object_name: str, payload: list, external_id_field: str='Id', batch_size: int=10000, use_serial: bool=False, salesforce_conn_id: str='salesforce_default', **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.operation = operation
        self.object_name = object_name
        self.payload = payload
        self.external_id_field = external_id_field
        self.batch_size = batch_size
        self.use_serial = use_serial
        self.salesforce_conn_id = salesforce_conn_id
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self.object_name:
            raise ValueError("The required parameter 'object_name' cannot have an empty value.")
        if self.operation not in self.available_operations:
            raise ValueError(f'Operation {self.operation!r} not found! Available operations are {self.available_operations}.')

    def execute(self, context: Context):
        if False:
            i = 10
            return i + 15
        '\n        Makes an HTTP request to Salesforce Bulk API.\n\n        :param context: The task context during execution.\n        :return: API response if do_xcom_push is True\n        '
        sf_hook = SalesforceHook(salesforce_conn_id=self.salesforce_conn_id)
        conn = sf_hook.get_conn()
        result = []
        if self.operation == 'insert':
            result = conn.bulk.__getattr__(self.object_name).insert(data=self.payload, batch_size=self.batch_size, use_serial=self.use_serial)
        elif self.operation == 'update':
            result = conn.bulk.__getattr__(self.object_name).update(data=self.payload, batch_size=self.batch_size, use_serial=self.use_serial)
        elif self.operation == 'upsert':
            result = conn.bulk.__getattr__(self.object_name).upsert(data=self.payload, external_id_field=self.external_id_field, batch_size=self.batch_size, use_serial=self.use_serial)
        elif self.operation == 'delete':
            result = conn.bulk.__getattr__(self.object_name).delete(data=self.payload, batch_size=self.batch_size, use_serial=self.use_serial)
        elif self.operation == 'hard_delete':
            result = conn.bulk.__getattr__(self.object_name).hard_delete(data=self.payload, batch_size=self.batch_size, use_serial=self.use_serial)
        if self.do_xcom_push and result:
            return result
        return None