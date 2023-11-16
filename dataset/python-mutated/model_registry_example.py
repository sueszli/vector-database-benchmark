import wandb
from dagster import AssetIn, Config, asset
from wandb import Artifact
MODEL_NAME = 'my_model'

@asset(name=MODEL_NAME, compute_kind='wandb')
def write_model() -> Artifact:
    if False:
        print('Hello World!')
    "Write your model.\n\n    Here, we have we're creating a very simple Artifact with the integration.\n\n    In a real scenario this would be more complex.\n\n    Returns:\n        wandb.Artifact: Our model\n    "
    return wandb.Artifact(MODEL_NAME, 'model')

class PromoteBestModelToProductionConfig(Config):
    model_registry: str

@asset(compute_kind='wandb', name='registered-model', ins={'artifact': AssetIn(key=MODEL_NAME)}, output_required=False)
def promote_best_model_to_production(artifact: Artifact, config: PromoteBestModelToProductionConfig):
    if False:
        while True:
            i = 10
    'Example that links a model stored in a W&B Artifact to the Model Registry.\n\n    Args:\n        context (AssetExecutionContext): Dagster execution context\n        artifact (wandb.wandb_sdk.wandb_artifacts.Artifact): Downloaded Artifact object\n    '
    performance_is_better = True
    if performance_is_better:
        model_registry = config.model_registry
        artifact.link(target_path=model_registry, aliases=['production'])