from dagster import job, make_values_resource
from dagster_wandb.launch.ops import run_launch_job
from dagster_wandb.resources import wandb_resource

@job(resource_defs={'wandb_config': make_values_resource(entity=str, project=str), 'wandb_resource': wandb_resource.configured({'api_key': {'env': 'WANDB_API_KEY'}})})
def run_launch_job_example():
    if False:
        i = 10
        return i + 15
    "Example of a simple Dagster job that runs a W&B Launch job.\n\n    In this example, we use a local Launch queue running inside the Dagster cluster.\n\n    You will have to first run the 'run_launch_agent_example' Dagster job.\n\n    You can also use deployed agent on Kubernetes, Sagemaker, and more.\n\n    Check the content of the config.yaml file to view the provided config.\n\n    Reference: https://docs.wandb.ai/guides/launch/agents\n    "
    run_launch_job.alias('my_launched_job')()