from typing import List, Mapping, Optional
from airflow.models.connection import Connection
from airflow.models.dag import DAG
from dagster import GraphDefinition, JobDefinition, ResourceDefinition, _check as check
from dagster._core.definitions.utils import validate_tags
from dagster._core.instance import IS_AIRFLOW_INGEST_PIPELINE_STR
from dagster_airflow.airflow_dag_converter import get_graph_definition_args
from dagster_airflow.resources import make_ephemeral_airflow_db_resource as make_ephemeral_airflow_db_resource
from dagster_airflow.utils import normalized_name

def make_dagster_job_from_airflow_dag(dag: DAG, tags: Optional[Mapping[str, str]]=None, connections: Optional[List[Connection]]=None, resource_defs: Optional[Mapping[str, ResourceDefinition]]={}) -> JobDefinition:
    if False:
        i = 10
        return i + 15
    "Construct a Dagster job corresponding to a given Airflow DAG.\n\n    Tasks in the resulting job will execute the ``execute()`` method on the corresponding\n    Airflow Operator. Dagster, any dependencies required by Airflow Operators, and the module\n    containing your DAG definition must be available in the Python environment within which your\n    Dagster solids execute.\n\n    To set Airflow's ``execution_date`` for use with Airflow Operator's ``execute()`` methods,\n    either:\n\n    1. (Best for ad hoc runs) Execute job directly. This will set execution_date to the\n        time (in UTC) of the run.\n\n    2. Add ``{'airflow_execution_date': utc_date_string}`` to the job tags. This will override\n        behavior from (1).\n\n        .. code-block:: python\n\n            my_dagster_job = make_dagster_job_from_airflow_dag(\n                    dag=dag,\n                    tags={'airflow_execution_date': utc_execution_date_str}\n            )\n            my_dagster_job.execute_in_process()\n\n    3. (Recommended) Add ``{'airflow_execution_date': utc_date_string}`` to the run tags,\n        such as in the Dagster UI. This will override behavior from (1) and (2)\n\n\n    We apply normalized_name() to the dag id and task ids when generating job name and op\n    names to ensure that names conform to Dagster's naming conventions.\n\n    Args:\n        dag (DAG): The Airflow DAG to compile into a Dagster job\n        tags (Dict[str, Field]): Job tags. Optionally include\n            `tags={'airflow_execution_date': utc_date_string}` to specify execution_date used within\n            execution of Airflow Operators.\n        connections (List[Connection]): List of Airflow Connections to be created in the Ephemeral\n            Airflow DB, if use_emphemeral_airflow_db is False this will be ignored.\n\n    Returns:\n        JobDefinition: The generated Dagster job\n\n    "
    check.inst_param(dag, 'dag', DAG)
    tags = check.opt_mapping_param(tags, 'tags')
    connections = check.opt_list_param(connections, 'connections', of_type=Connection)
    mutated_tags = dict(tags)
    if IS_AIRFLOW_INGEST_PIPELINE_STR not in tags:
        mutated_tags[IS_AIRFLOW_INGEST_PIPELINE_STR] = 'true'
    mutated_tags = validate_tags(mutated_tags)
    (node_dependencies, node_defs) = get_graph_definition_args(dag=dag)
    graph_def = GraphDefinition(name=normalized_name(dag.dag_id), description='', node_defs=node_defs, dependencies=node_dependencies, tags=mutated_tags)
    if resource_defs is None or 'airflow_db' not in resource_defs:
        resource_defs = dict(resource_defs) if resource_defs else {}
        resource_defs['airflow_db'] = make_ephemeral_airflow_db_resource(connections=connections)
    job_def = JobDefinition(name=normalized_name(dag.dag_id), description='', graph_def=graph_def, resource_defs=resource_defs, tags=mutated_tags, metadata={}, op_retry_policy=None, version_strategy=None)
    return job_def