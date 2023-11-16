from __future__ import annotations
import sys
from typing import Any
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

def wait_for_extended_operation(operation: ExtendedOperation, verbose_name: str='operation', timeout: int=300) -> Any:
    if False:
        for i in range(10):
            print('nop')
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

def start_instance_with_encryption_key(project_id: str, zone: str, instance_name: str, key: bytes) -> None:
    if False:
        return 10
    '\n    Starts a stopped Google Compute Engine instance (with encrypted disks).\n    Args:\n        project_id: project ID or project number of the Cloud project your instance belongs to.\n        zone: name of the zone your instance belongs to.\n        instance_name: name of the instance your want to start.\n        key: bytes object representing a raw base64 encoded key to your machines boot disk.\n            For more information about disk encryption see:\n            https://cloud.google.com/compute/docs/disks/customer-supplied-encryption#specifications\n    '
    instance_client = compute_v1.InstancesClient()
    instance_data = instance_client.get(project=project_id, zone=zone, instance=instance_name)
    disk_data = compute_v1.CustomerEncryptionKeyProtectedDisk()
    disk_data.source = instance_data.disks[0].source
    disk_data.disk_encryption_key = compute_v1.CustomerEncryptionKey()
    disk_data.disk_encryption_key.raw_key = key
    enc_data = compute_v1.InstancesStartWithEncryptionKeyRequest()
    enc_data.disks = [disk_data]
    operation = instance_client.start_with_encryption_key(project=project_id, zone=zone, instance=instance_name, instances_start_with_encryption_key_request_resource=enc_data)
    wait_for_extended_operation(operation, 'instance start (with encrypted disk)')