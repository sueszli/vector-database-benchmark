from typing import Iterator
from third_party_api import is_external_process_done, launch_external_process
from dagster import AssetExecutionContext, PipesResult, PipesTempFileContextInjector, PipesTempFileMessageReader, asset, open_pipes_session

@asset
def some_pipes_asset(context: AssetExecutionContext) -> Iterator[PipesResult]:
    if False:
        print('Hello World!')
    with open_pipes_session(context=context, extras={'foo': 'bar'}, context_injector=PipesTempFileContextInjector(), message_reader=PipesTempFileMessageReader()) as pipes_session:
        env_vars = pipes_session.get_bootstrap_env_vars()
        external_process = launch_external_process(env_vars)
        while not is_external_process_done(external_process):
            yield from pipes_session.get_results()
    yield from pipes_session.get_results()