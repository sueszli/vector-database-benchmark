from __future__ import annotations
import sys
from typing import Any
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

def wait_for_extended_operation(operation: ExtendedOperation, verbose_name: str='operation', timeout: int=300) -> Any:
    if False:
        print('Hello World!')
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

def set_delete_protection(project_id: str, zone: str, instance_name: str, delete_protection: bool) -> None:
    if False:
        return 10
    '\n    Updates the delete protection setting of given instance.\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone you want to use. For example: “us-west3-b”\n        instance_name: name of the instance to update.\n        delete_protection: boolean value indicating if the virtual machine should be\n            protected against deletion or not.\n    '
    instance_client = compute_v1.InstancesClient()
    request = compute_v1.SetDeletionProtectionInstanceRequest()
    request.project = project_id
    request.zone = zone
    request.resource = instance_name
    request.deletion_protection = delete_protection
    operation = instance_client.set_deletion_protection(request)
    wait_for_extended_operation(operation, 'changing delete protection setting')