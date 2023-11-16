from dagster import job, make_values_resource
from dagster_wandb.launch.ops import run_launch_agent
from dagster_wandb.resources import wandb_resource

@job(resource_defs={'wandb_config': make_values_resource(entity=str, project=str), 'wandb_resource': wandb_resource.configured({'api_key': {'env': 'WANDB_API_KEY'}})})
def run_launch_agent_example():
    if False:
        while True:
            i = 10
    'Example of a simple job that runs a W&B Launch agent.\n\n    The Launch agent will run until stopped.\n\n    Check the content of the config.yaml file to view the provided config.\n\n    Agents are processes that poll Launch queues and execute the jobs (or dispatch them to external\n    services to be executed) in order.\n\n    Reference: https://docs.wandb.ai/guides/launch/agents\n    '
    run_launch_agent()