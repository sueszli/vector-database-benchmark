"""This module contains an operator to move data from Hive to Samba."""
from __future__ import annotations
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Sequence
from airflow.models import BaseOperator
from airflow.providers.apache.hive.hooks.hive import HiveServer2Hook
from airflow.providers.samba.hooks.samba import SambaHook
from airflow.utils.operator_helpers import context_to_airflow_vars
if TYPE_CHECKING:
    from airflow.utils.context import Context

class HiveToSambaOperator(BaseOperator):
    """
    Execute hql code in a specific Hive database and load the results as a csv to a Samba location.

    :param hql: the hql to be exported. (templated)
    :param destination_filepath: the file path to where the file will be pushed onto samba
    :param samba_conn_id: reference to the samba destination
    :param hiveserver2_conn_id: Reference to the
        :ref: `Hive Server2 thrift service connection id <howto/connection:hiveserver2>`.
    """
    template_fields: Sequence[str] = ('hql', 'destination_filepath')
    template_ext: Sequence[str] = ('.hql', '.sql')
    template_fields_renderers = {'hql': 'hql'}

    def __init__(self, *, hql: str, destination_filepath: str, samba_conn_id: str='samba_default', hiveserver2_conn_id: str='hiveserver2_default', **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.hiveserver2_conn_id = hiveserver2_conn_id
        self.samba_conn_id = samba_conn_id
        self.destination_filepath = destination_filepath
        self.hql = hql.strip().rstrip(';')

    def execute(self, context: Context):
        if False:
            print('Hello World!')
        with NamedTemporaryFile() as tmp_file:
            self.log.info('Fetching file from Hive')
            hive = HiveServer2Hook(hiveserver2_conn_id=self.hiveserver2_conn_id)
            hive.to_csv(self.hql, csv_filepath=tmp_file.name, hive_conf=context_to_airflow_vars(context))
            self.log.info('Pushing to samba')
            samba = SambaHook(samba_conn_id=self.samba_conn_id)
            samba.push_from_local(self.destination_filepath, tmp_file.name)