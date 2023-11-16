import os
import sys
from dagster_databricks import PipesDbfsContextInjector, PipesDbfsMessageReader
from dagster_databricks.pipes import PipesDbfsLogReader
from dagster import AssetExecutionContext, asset, open_pipes_session
from databricks.sdk import WorkspaceClient

@asset
def databricks_asset(context: AssetExecutionContext):
    if False:
        while True:
            i = 10
    client = WorkspaceClient(host=os.environ['DATABRICKS_HOST'], token=os.environ['DATABRICKS_TOKEN'])
    extras = {'sample_rate': 1.0}
    with open_pipes_session(context=context, extras=extras, context_injector=PipesDbfsContextInjector(client=client), message_reader=PipesDbfsMessageReader(client=client, log_readers=[PipesDbfsLogReader(client=client, remote_log_name='stdout', target_stream=sys.stdout), PipesDbfsLogReader(client=client, remote_log_name='stderr', target_stream=sys.stderr)])) as pipes_session:
        env_vars = pipes_session.get_bootstrap_env_vars()
        custom_databricks_launch_code(env_vars)
        yield from custom_databricks_launch_code(pipes_session)
    yield from pipes_session.get_results()