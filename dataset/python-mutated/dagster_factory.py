import os
from typing import List, Mapping, Optional, Tuple
from airflow.models.connection import Connection
from airflow.models.dagbag import DagBag
from dagster import Definitions, JobDefinition, ResourceDefinition, ScheduleDefinition, _check as check
from dagster_airflow.dagster_job_factory import make_dagster_job_from_airflow_dag
from dagster_airflow.dagster_schedule_factory import _is_dag_is_schedule, make_dagster_schedule_from_airflow_dag
from dagster_airflow.patch_airflow_example_dag import patch_airflow_example_dag
from dagster_airflow.resources import make_ephemeral_airflow_db_resource as make_ephemeral_airflow_db_resource
from dagster_airflow.resources.airflow_ephemeral_db import AirflowEphemeralDatabase
from dagster_airflow.resources.airflow_persistent_db import AirflowPersistentDatabase
from dagster_airflow.utils import is_airflow_2_loaded_in_environment

def make_dagster_definitions_from_airflow_dag_bag(dag_bag: DagBag, connections: Optional[List[Connection]]=None, resource_defs: Optional[Mapping[str, ResourceDefinition]]={}) -> Definitions:
    if False:
        return 10
    'Construct a Dagster definition corresponding to Airflow DAGs in DagBag.\n\n    Usage:\n        Create `make_dagster_definition.py`:\n            from dagster_airflow import make_dagster_definition_from_airflow_dag_bag\n            from airflow_home import my_dag_bag\n\n            def make_definition_from_dag_bag():\n                return make_dagster_definition_from_airflow_dag_bag(my_dag_bag)\n\n        Use Definitions as usual, for example:\n            `dagster-webserver -f path/to/make_dagster_definition.py`\n\n    Args:\n        dag_bag (DagBag): Airflow DagBag Model\n        connections (List[Connection]): List of Airflow Connections to be created in the Airflow DB\n\n    Returns:\n        Definitions\n    '
    check.inst_param(dag_bag, 'dag_bag', DagBag)
    connections = check.opt_list_param(connections, 'connections', of_type=Connection)
    resource_defs = check.opt_mapping_param(resource_defs, 'resource_defs')
    if resource_defs is None or 'airflow_db' not in resource_defs:
        resource_defs = dict(resource_defs) if resource_defs else {}
        resource_defs['airflow_db'] = make_ephemeral_airflow_db_resource(connections=connections)
    (schedules, jobs) = make_schedules_and_jobs_from_airflow_dag_bag(dag_bag=dag_bag, connections=connections, resource_defs=resource_defs)
    return Definitions(schedules=schedules, jobs=jobs, resources=resource_defs)

def make_dagster_definitions_from_airflow_dags_path(dag_path: str, safe_mode: bool=True, connections: Optional[List[Connection]]=None, resource_defs: Optional[Mapping[str, ResourceDefinition]]={}) -> Definitions:
    if False:
        while True:
            i = 10
    "Construct a Dagster repository corresponding to Airflow DAGs in dag_path.\n\n    Usage:\n        Create ``make_dagster_definitions.py``:\n\n        .. code-block:: python\n\n            from dagster_airflow import make_dagster_definitions_from_airflow_dags_path\n\n            def make_definitions_from_dir():\n                return make_dagster_definitions_from_airflow_dags_path(\n                    '/path/to/dags/',\n                )\n\n        Use RepositoryDefinition as usual, for example:\n        ``dagster-webserver -f path/to/make_dagster_repo.py -n make_repo_from_dir``\n\n    Args:\n        dag_path (str): Path to directory or file that contains Airflow Dags\n        include_examples (bool): True to include Airflow's example DAGs. (default: False)\n        safe_mode (bool): True to use Airflow's default heuristic to find files that contain DAGs\n            (ie find files that contain both b'DAG' and b'airflow') (default: True)\n        connections (List[Connection]): List of Airflow Connections to be created in the Airflow DB\n\n    Returns:\n        Definitions\n    "
    check.str_param(dag_path, 'dag_path')
    check.bool_param(safe_mode, 'safe_mode')
    connections = check.opt_list_param(connections, 'connections', of_type=Connection)
    resource_defs = check.opt_mapping_param(resource_defs, 'resource_defs')
    if resource_defs is None or 'airflow_db' not in resource_defs:
        resource_defs = dict(resource_defs) if resource_defs else {}
        resource_defs['airflow_db'] = make_ephemeral_airflow_db_resource(connections=connections)
    if resource_defs['airflow_db'].resource_fn.__qualname__.split('.')[0] == 'AirflowEphemeralDatabase':
        AirflowEphemeralDatabase._initialize_database(connections=connections)
    elif resource_defs['airflow_db'].resource_fn.__qualname__.split('.')[0] == 'AirflowPersistentDatabase':
        AirflowPersistentDatabase._initialize_database(uri=os.getenv('AIRFLOW__DATABASE__SQL_ALCHEMY_CONN', '') if is_airflow_2_loaded_in_environment() else os.getenv('AIRFLOW__CORE__SQL_ALCHEMY_CONN', ''), connections=connections)
    dag_bag = DagBag(dag_folder=dag_path, include_examples=False, safe_mode=safe_mode)
    return make_dagster_definitions_from_airflow_dag_bag(dag_bag=dag_bag, connections=connections, resource_defs=resource_defs)

def make_dagster_definitions_from_airflow_example_dags(resource_defs: Optional[Mapping[str, ResourceDefinition]]={}) -> Definitions:
    if False:
        return 10
    "Construct a Dagster repository for Airflow's example DAGs.\n\n    Usage:\n\n        Create `make_dagster_definitions.py`:\n            from dagster_airflow import make_dagster_definitions_from_airflow_example_dags\n\n            def make_airflow_example_dags():\n                return make_dagster_definitions_from_airflow_example_dags()\n\n        Use Definitions as usual, for example:\n            `dagster-webserver -f path/to/make_dagster_definitions.py`\n\n    Args:\n        resource_defs: Optional[Mapping[str, ResourceDefinition]]\n            Resource definitions to be used with the definitions\n\n    Returns:\n        Definitions\n    "
    dag_bag = DagBag(dag_folder='some/empty/folder/with/no/dags', include_examples=True)
    patch_airflow_example_dag(dag_bag)
    return make_dagster_definitions_from_airflow_dag_bag(dag_bag=dag_bag, resource_defs=resource_defs)

def make_schedules_and_jobs_from_airflow_dag_bag(dag_bag: DagBag, connections: Optional[List[Connection]]=None, resource_defs: Optional[Mapping[str, ResourceDefinition]]={}) -> Tuple[List[ScheduleDefinition], List[JobDefinition]]:
    if False:
        while True:
            i = 10
    'Construct Dagster Schedules and Jobs corresponding to Airflow DagBag.\n\n    Args:\n        dag_bag (DagBag): Airflow DagBag Model\n        connections (List[Connection]): List of Airflow Connections to be created in the Airflow DB\n\n    Returns:\n        - List[ScheduleDefinition]: The generated Dagster Schedules\n        - List[JobDefinition]: The generated Dagster Jobs\n    '
    check.inst_param(dag_bag, 'dag_bag', DagBag)
    connections = check.opt_list_param(connections, 'connections', of_type=Connection)
    job_defs = []
    schedule_defs = []
    count = 0
    sorted_dag_ids = sorted(dag_bag.dag_ids)
    for dag_id in sorted_dag_ids:
        dag = dag_bag.dags.get(dag_id)
        if not dag:
            continue
        if _is_dag_is_schedule(dag):
            schedule_defs.append(make_dagster_schedule_from_airflow_dag(dag=dag, tags=None, connections=connections, resource_defs=resource_defs))
        else:
            job_defs.append(make_dagster_job_from_airflow_dag(dag=dag, tags=None, connections=connections, resource_defs=resource_defs))
        count += 1
    return (schedule_defs, job_defs)