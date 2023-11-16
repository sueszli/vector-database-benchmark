"""Example DAG demonstrating the usage of dynamic task mapping with non-TaskFlow operators."""
from __future__ import annotations
from datetime import datetime
from airflow.models.baseoperator import BaseOperator
from airflow.models.dag import DAG

class AddOneOperator(BaseOperator):
    """A custom operator that adds one to the input."""

    def __init__(self, value, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.value = value

    def execute(self, context):
        if False:
            print('Hello World!')
        return self.value + 1

class SumItOperator(BaseOperator):
    """A custom operator that sums the input."""
    template_fields = ('values',)

    def __init__(self, values, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.values = values

    def execute(self, context):
        if False:
            return 10
        total = sum(self.values)
        print(f'Total was {total}')
        return total
with DAG(dag_id='example_dynamic_task_mapping_with_no_taskflow_operators', start_date=datetime(2022, 3, 4), catchup=False):
    add_one_task = AddOneOperator.partial(task_id='add_one').expand(value=[1, 2, 3])
    sum_it_task = SumItOperator(task_id='sum_it', values=add_one_task.output)