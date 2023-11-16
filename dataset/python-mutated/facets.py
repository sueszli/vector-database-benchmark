from __future__ import annotations
from attrs import define
from openlineage.client.facet import BaseFacet
from openlineage.client.utils import RedactMixin

@define(slots=False)
class AirflowMappedTaskRunFacet(BaseFacet):
    """Run facet containing information about mapped tasks."""
    mapIndex: int
    operatorClass: str
    _additional_skip_redact: list[str] = ['operatorClass']

    @classmethod
    def from_task_instance(cls, task_instance):
        if False:
            print('Hello World!')
        task = task_instance.task
        from airflow.providers.openlineage.utils.utils import get_operator_class
        return cls(mapIndex=task_instance.map_index, operatorClass=f'{get_operator_class(task).__module__}.{get_operator_class(task).__name__}')

@define(slots=False)
class AirflowRunFacet(BaseFacet):
    """Composite Airflow run facet."""
    dag: dict
    dagRun: dict
    task: dict
    taskInstance: dict
    taskUuid: str

@define(slots=False)
class UnknownOperatorInstance(RedactMixin):
    """Describes an unknown operator.

    This specifies the (class) name of the operator and its properties.
    """
    name: str
    properties: dict[str, object]
    type: str = 'operator'
    _skip_redact: list[str] = ['name', 'type']

@define(slots=False)
class UnknownOperatorAttributeRunFacet(BaseFacet):
    """RunFacet that describes unknown operators in an Airflow DAG."""
    unknownItems: list[UnknownOperatorInstance]