"""
AWS ECS Executor configuration.

This is the configuration for calling the ECS ``run_task`` function. The AWS ECS Executor calls
Boto3's ``run_task(**kwargs)`` function with the kwargs templated by this dictionary. See the URL
below for documentation on the parameters accepted by the Boto3 run_task function.

.. seealso::
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ecs.html#ECS.Client.run_task

"""
from __future__ import annotations
import json
from json import JSONDecodeError
from airflow.configuration import conf
from airflow.providers.amazon.aws.executors.ecs.utils import CONFIG_GROUP_NAME, AllEcsConfigKeys, RunTaskKwargsConfigKeys, camelize_dict_keys, parse_assign_public_ip
from airflow.utils.helpers import prune_dict

def _fetch_templated_kwargs() -> dict[str, str]:
    if False:
        while True:
            i = 10
    run_task_kwargs_value = conf.get(CONFIG_GROUP_NAME, AllEcsConfigKeys.RUN_TASK_KWARGS, fallback=dict())
    return json.loads(str(run_task_kwargs_value))

def _fetch_config_values() -> dict[str, str]:
    if False:
        print('Hello World!')
    return prune_dict({key: conf.get(CONFIG_GROUP_NAME, key, fallback=None) for key in RunTaskKwargsConfigKeys()})

def build_task_kwargs() -> dict:
    if False:
        i = 10
        return i + 15
    task_kwargs = _fetch_config_values()
    task_kwargs.update(_fetch_templated_kwargs())
    task_kwargs['count'] = 1
    if 'overrides' not in task_kwargs:
        task_kwargs['overrides'] = {}
    if 'containerOverrides' not in task_kwargs['overrides']:
        task_kwargs['overrides']['containerOverrides'] = [{}]
    task_kwargs['overrides']['containerOverrides'][0]['name'] = task_kwargs.pop(AllEcsConfigKeys.CONTAINER_NAME)
    task_kwargs['overrides']['containerOverrides'][0]['command'] = []
    if any([(subnets := task_kwargs.pop(AllEcsConfigKeys.SUBNETS, None)), (security_groups := task_kwargs.pop(AllEcsConfigKeys.SECURITY_GROUPS, None)), (assign_public_ip := task_kwargs.pop(AllEcsConfigKeys.ASSIGN_PUBLIC_IP, None)) is not None]):
        network_config = prune_dict({'awsvpcConfiguration': {'subnets': str(subnets).split(',') if subnets else None, 'securityGroups': str(security_groups).split(',') if security_groups else None, 'assignPublicIp': parse_assign_public_ip(assign_public_ip)}})
        if 'subnets' not in network_config['awsvpcConfiguration']:
            raise ValueError('At least one subnet is required to run a task.')
        task_kwargs['networkConfiguration'] = network_config
    task_kwargs = camelize_dict_keys(task_kwargs)
    try:
        json.loads(json.dumps(task_kwargs))
    except JSONDecodeError:
        raise ValueError('AWS ECS Executor config values must be JSON serializable.')
    return task_kwargs