from __future__ import annotations
import sys
from typing import Any
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

def wait_for_extended_operation(operation: ExtendedOperation, verbose_name: str='operation', timeout: int=300) -> Any:
    if False:
        i = 10
        return i + 15
    '\n    Waits for the extended (long-running) operation to complete.\n\n    If the operation is successful, it will return its result.\n    If the operation ends with an error, an exception will be raised.\n    If there were any warnings during the execution of the operation\n    they will be printed to sys.stderr.\n\n    Args:\n        operation: a long-running operation you want to wait on.\n        verbose_name: (optional) a more verbose name of the operation,\n            used only during error and warning reporting.\n        timeout: how long (in seconds) to wait for operation to finish.\n            If None, wait indefinitely.\n\n    Returns:\n        Whatever the operation.result() returns.\n\n    Raises:\n        This method will raise the exception received from `operation.exception()`\n        or RuntimeError if there is no exception set, but there is an `error_code`\n        set for the `operation`.\n\n        In case of an operation taking longer than `timeout` seconds to complete,\n        a `concurrent.futures.TimeoutError` will be raised.\n    '
    result = operation.result(timeout=timeout)
    if operation.error_code:
        print(f'Error during {verbose_name}: [Code: {operation.error_code}]: {operation.error_message}', file=sys.stderr, flush=True)
        print(f'Operation ID: {operation.name}', file=sys.stderr, flush=True)
        raise operation.exception() or RuntimeError(operation.error_message)
    if operation.warnings:
        print(f'Warnings during {verbose_name}:\n', file=sys.stderr, flush=True)
        for warning in operation.warnings:
            print(f' - {warning.code}: {warning.message}', file=sys.stderr, flush=True)
    return result

def create_instance_from_template(project_id: str, zone: str, instance_name: str, instance_template_url: str) -> compute_v1.Instance:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a Compute Engine VM instance from an instance template.\n\n    Args:\n        project_id: ID or number of the project you want to use.\n        zone: Name of the zone you want to check, for example: us-west3-b\n        instance_name: Name of the new instance.\n        instance_template_url: URL of the instance template used for creating the new instance.\n            It can be a full or partial URL.\n            Examples:\n            - https://www.googleapis.com/compute/v1/projects/project/global/instanceTemplates/example-instance-template\n            - projects/project/global/instanceTemplates/example-instance-template\n            - global/instanceTemplates/example-instance-template\n\n    Returns:\n        Instance object.\n    '
    instance_client = compute_v1.InstancesClient()
    instance_insert_request = compute_v1.InsertInstanceRequest()
    instance_insert_request.project = project_id
    instance_insert_request.zone = zone
    instance_insert_request.source_instance_template = instance_template_url
    instance_insert_request.instance_resource.name = instance_name
    operation = instance_client.insert(instance_insert_request)
    wait_for_extended_operation(operation, 'instance creation')
    return instance_client.get(project=project_id, zone=zone, instance=instance_name)