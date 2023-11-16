"""jsonschema for validating serialized DAG and operator."""
from __future__ import annotations
import pkgutil
from typing import TYPE_CHECKING, Iterable
from airflow.exceptions import AirflowException
from airflow.settings import json
from airflow.typing_compat import Protocol
if TYPE_CHECKING:
    import jsonschema

class Validator(Protocol):
    """
    This class is only used for type checking.

    A workaround for IDEs, mypy, etc. due to the way ``Draft7Validator`` is created.
    They are created or do not inherit from proper classes.
    Hence, you can not have ``type: Draft7Validator``.
    """

    def is_valid(self, instance) -> bool:
        if False:
            while True:
                i = 10
        'Check if the instance is valid under the current schema.'
        ...

    def validate(self, instance) -> None:
        if False:
            while True:
                i = 10
        'Check if the instance is valid under the current schema, raising validation error if not.'
        ...

    def iter_errors(self, instance) -> Iterable[jsonschema.exceptions.ValidationError]:
        if False:
            for i in range(10):
                print('nop')
        'Lazily yield each of the validation errors in the given instance.'
        ...

def load_dag_schema_dict() -> dict:
    if False:
        for i in range(10):
            print('nop')
    'Load & return Json Schema for DAG as Python dict.'
    schema_file_name = 'schema.json'
    schema_file = pkgutil.get_data(__name__, schema_file_name)
    if schema_file is None:
        raise AirflowException(f'Schema file {schema_file_name} does not exists')
    schema = json.loads(schema_file.decode())
    return schema

def load_dag_schema() -> Validator:
    if False:
        for i in range(10):
            print('nop')
    'Load & Validate Json Schema for DAG.'
    import jsonschema
    schema = load_dag_schema_dict()
    return jsonschema.Draft7Validator(schema)