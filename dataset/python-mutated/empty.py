from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.models.baseoperator import BaseOperator
if TYPE_CHECKING:
    from airflow.utils.context import Context

class EmptyOperator(BaseOperator):
    """
    Operator that does literally nothing.

    It can be used to group tasks in a DAG.
    The task is evaluated by the scheduler but never processed by the executor.
    """
    ui_color = '#e8f7e4'
    inherits_from_empty_operator = True

    def execute(self, context: Context):
        if False:
            i = 10
            return i + 15
        pass