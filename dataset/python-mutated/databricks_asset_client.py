import os
import sys
from dagster_databricks import PipesDatabricksClient
from dagster import AssetExecutionContext, Definitions, EnvVar, asset
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs

@asset
def databricks_asset(context: AssetExecutionContext, pipes_databricks: PipesDatabricksClient):
    if False:
        for i in range(10):
            print('nop')
    task = jobs.SubmitTask.from_dict({'new_cluster': {'spark_version': '12.2.x-scala2.12', 'node_type_id': 'i3.xlarge', 'num_workers': 0, 'cluster_log_conf': {'dbfs': {'destination': 'dbfs:/cluster-logs-dir-noexist'}}}, 'libraries': [{'pypi': {'package': 'dagster-pipes'}}], 'task_key': 'some-key', 'spark_python_task': {'python_file': 'dbfs:/my_python_script.py', 'source': jobs.Source.WORKSPACE}})
    print('This will be forwarded back to Dagster stdout')
    print('This will be forwarded back to Dagster stderr', file=sys.stderr)
    extras = {'some_parameter': 100}
    return pipes_databricks.run(task=task, context=context, extras=extras).get_materialize_result()
pipes_databricks_resource = PipesDatabricksClient(client=WorkspaceClient(host=os.getenv('DATABRICKS_HOST'), token=os.getenv('DATABRICKS_TOKEN')))
defs = Definitions(assets=[databricks_asset], resources={'pipes_databricks': pipes_databricks_resource})