from __future__ import annotations
import sys
from typing import Any
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

def wait_for_extended_operation(operation: ExtendedOperation, verbose_name: str='operation', timeout: int=300) -> Any:
    if False:
        while True:
            i = 10
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

def change_machine_type(project_id: str, zone: str, instance_name: str, new_machine_type: str) -> None:
    if False:
        return 10
    "\n    Changes the machine type of VM. The VM needs to be in the 'TERMINATED' state for this operation to be successful.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone your instance belongs to.\n        instance_name: name of the VM you want to modify.\n        new_machine_type: the new machine type you want to use for the VM.\n            For example: `e2-standard-8`, `e2-custom-4-2048` or `m1-ultramem-40`\n            More about machine types: https://cloud.google.com/compute/docs/machine-resource\n    "
    client = compute_v1.InstancesClient()
    instance = client.get(project=project_id, zone=zone, instance=instance_name)
    if instance.status != compute_v1.Instance.Status.TERMINATED.name:
        raise RuntimeError(f'Only machines in TERMINATED state can have their machine type changed. {instance.name} is in {instance.status}({instance.status_message}) state.')
    machine_type = compute_v1.InstancesSetMachineTypeRequest()
    machine_type.machine_type = f'projects/{project_id}/zones/{zone}/machineTypes/{new_machine_type}'
    operation = client.set_machine_type(project=project_id, zone=zone, instance=instance_name, instances_set_machine_type_request_resource=machine_type)
    wait_for_extended_operation(operation, 'changing machine type')