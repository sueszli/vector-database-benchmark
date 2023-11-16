from __future__ import annotations
from marshmallow import fields, validate
from airflow.utils.state import State

class DagStateField(fields.String):
    """Schema for DagState Enum."""

    def __init__(self, **metadata):
        if False:
            print('Hello World!')
        super().__init__(**metadata)
        self.validators = [validate.OneOf(State.dag_states), *self.validators]

class TaskInstanceStateField(fields.String):
    """Schema for TaskInstanceState Enum."""

    def __init__(self, **metadata):
        if False:
            i = 10
            return i + 15
        super().__init__(**metadata)
        self.validators = [validate.OneOf(State.task_states), *self.validators]