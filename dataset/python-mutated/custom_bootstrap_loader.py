from cloud_service import METADATA
from dagster_pipes import DAGSTER_PIPES_CONTEXT_ENV_VAR, DAGSTER_PIPES_MESSAGES_ENV_VAR, PipesParams, PipesParamsLoader

class MyCustomParamsLoader(PipesParamsLoader):

    def is_dagster_pipes_process(self) -> bool:
        if False:
            return 10
        return DAGSTER_PIPES_CONTEXT_ENV_VAR in METADATA

    def load_context_params(self) -> PipesParams:
        if False:
            print('Hello World!')
        return METADATA[DAGSTER_PIPES_CONTEXT_ENV_VAR]

    def load_messages_params(self) -> PipesParams:
        if False:
            i = 10
            return i + 15
        return METADATA[DAGSTER_PIPES_MESSAGES_ENV_VAR]