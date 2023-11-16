from __future__ import annotations
import sys
import time
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

def add_extended_memory_to_instance(project_id: str, zone: str, instance_name: str, new_memory: int):
    if False:
        return 10
    '\n    Modify an existing VM to use extended memory.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone to create the instance in. For example: "us-west3-b"\n        instance_name: name of the new virtual machine (VM) instance.\n        new_memory: the amount of memory for the VM instance, in megabytes.\n\n    Returns:\n        Instance object.\n    '
    instance_client = compute_v1.InstancesClient()
    instance = instance_client.get(project=project_id, zone=zone, instance=instance_name)
    if not ('n1-' in instance.machine_type or 'n2-' in instance.machine_type or 'n2d-' in instance.machine_type):
        raise RuntimeError('Extra memory is available only for N1, N2 and N2D CPUs.')
    if instance.status not in (instance.Status.TERMINATED.name, instance.Status.STOPPED.name):
        operation = instance_client.stop(project=project_id, zone=zone, instance=instance_name)
        wait_for_extended_operation(operation, 'instance stopping')
        start = time.time()
        while instance.status not in (instance.Status.TERMINATED.name, instance.Status.STOPPED.name):
            instance = instance_client.get(project=project_id, zone=zone, instance=instance_name)
            time.sleep(2)
            if time.time() - start >= 300:
                raise TimeoutError()
    (start, end) = instance.machine_type.rsplit('-', maxsplit=1)
    instance.machine_type = start + f'-{new_memory}-ext'
    operation = instance_client.update(project=project_id, zone=zone, instance=instance_name, instance_resource=instance)
    wait_for_extended_operation(operation, 'instance update')
    return instance_client.get(project=project_id, zone=zone, instance=instance_name)