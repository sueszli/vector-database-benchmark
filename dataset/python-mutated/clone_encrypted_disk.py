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

def create_disk_from_customer_encrypted_disk(project_id: str, zone: str, disk_name: str, disk_type: str, disk_size_gb: int, disk_link: str, encryption_key: bytes) -> compute_v1.Disk:
    if False:
        print('Hello World!')
    '\n    Creates a zonal non-boot persistent disk in a project with the copy of data from an existing disk.\n\n    The encryption key must be the same for the source disk and the new disk.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone in which you want to create the disk.\n        disk_name: name of the disk you want to create.\n        disk_type: the type of disk you want to create. This value uses the following format:\n            "zones/{zone}/diskTypes/(pd-standard|pd-ssd|pd-balanced|pd-extreme)".\n            For example: "zones/us-west3-b/diskTypes/pd-ssd"\n        disk_size_gb: size of the new disk in gigabytes\n        disk_link: a link to the disk you want to use as a source for the new disk.\n            This value uses the following format: "projects/{project_name}/zones/{zone}/disks/{disk_name}"\n        encryption_key: customer-supplied encryption key used for encrypting\n            data in the source disk. The data will be encrypted with the same key\n            in the new disk.\n\n    Returns:\n        An attachable copy of an existing disk.\n    '
    disk_client = compute_v1.DisksClient()
    disk = compute_v1.Disk()
    disk.zone = zone
    disk.size_gb = disk_size_gb
    disk.source_disk = disk_link
    disk.type_ = disk_type
    disk.name = disk_name
    disk.disk_encryption_key = compute_v1.CustomerEncryptionKey()
    disk.disk_encryption_key.raw_key = encryption_key
    operation = disk_client.insert(project=project_id, zone=zone, disk_resource=disk)
    wait_for_extended_operation(operation, 'disk creation')
    return disk_client.get(project=project_id, zone=zone, disk=disk_name)