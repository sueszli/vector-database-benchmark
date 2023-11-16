from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
from marshmallow import Schema, fields
from airflow.api_connexion.schemas.common_schema import ClassReferenceSchema, ColorField, TimeDeltaSchema, WeightRuleField
from airflow.api_connexion.schemas.dag_schema import DAGSchema
from airflow.models.mappedoperator import MappedOperator
if TYPE_CHECKING:
    from airflow.models.operator import Operator

class TaskSchema(Schema):
    """Task schema."""
    class_ref = fields.Method('_get_class_reference', dump_only=True)
    operator_name = fields.Method('_get_operator_name', dump_only=True)
    task_id = fields.String(dump_only=True)
    owner = fields.String(dump_only=True)
    start_date = fields.DateTime(dump_only=True)
    end_date = fields.DateTime(dump_only=True)
    trigger_rule = fields.String(dump_only=True)
    extra_links = fields.List(fields.Nested(ClassReferenceSchema), dump_only=True, attribute='operator_extra_links')
    depends_on_past = fields.Boolean(dump_only=True)
    wait_for_downstream = fields.Boolean(dump_only=True)
    retries = fields.Number(dump_only=True)
    queue = fields.String(dump_only=True)
    pool = fields.String(dump_only=True)
    pool_slots = fields.Number(dump_only=True)
    execution_timeout = fields.Nested(TimeDeltaSchema, dump_only=True)
    retry_delay = fields.Nested(TimeDeltaSchema, dump_only=True)
    retry_exponential_backoff = fields.Boolean(dump_only=True)
    priority_weight = fields.Number(dump_only=True)
    weight_rule = WeightRuleField(dump_only=True)
    ui_color = ColorField(dump_only=True)
    ui_fgcolor = ColorField(dump_only=True)
    template_fields = fields.List(fields.String(), dump_only=True)
    sub_dag = fields.Nested(DAGSchema, dump_only=True)
    downstream_task_ids = fields.List(fields.String(), dump_only=True)
    params = fields.Method('_get_params', dump_only=True)
    is_mapped = fields.Method('_get_is_mapped', dump_only=True)

    @staticmethod
    def _get_class_reference(obj):
        if False:
            print('Hello World!')
        result = ClassReferenceSchema().dump(obj)
        return result.data if hasattr(result, 'data') else result

    @staticmethod
    def _get_operator_name(obj):
        if False:
            return 10
        return obj.operator_name

    @staticmethod
    def _get_params(obj):
        if False:
            i = 10
            return i + 15
        'Get the Params defined in a Task.'
        params = obj.params
        return {k: v.dump() for (k, v) in params.items()}

    @staticmethod
    def _get_is_mapped(obj):
        if False:
            print('Hello World!')
        return isinstance(obj, MappedOperator)

class TaskCollection(NamedTuple):
    """List of Tasks with metadata."""
    tasks: list[Operator]
    total_entries: int

class TaskCollectionSchema(Schema):
    """Schema for TaskCollection."""
    tasks = fields.List(fields.Nested(TaskSchema))
    total_entries = fields.Int()
task_schema = TaskSchema()
task_collection_schema = TaskCollectionSchema()