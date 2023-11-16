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

def attach_disk(project_id: str, zone: str, instance_name: str, disk_link: str, mode: str) -> None:
    if False:
        print('Hello World!')
    '\n    Attaches a non-boot persistent disk to a specified compute instance. The disk might be zonal or regional.\n\n    You need following permissions to execute this action:\n    https://cloud.google.com/compute/docs/disks/regional-persistent-disk#expandable-1\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone:name of the zone in which the instance you want to use resides.\n        instance_name: name of the compute instance you want to attach a disk to.\n        disk_link: full or partial URL to a persistent disk that you want to attach. This can be either\n            regional or zonal disk.\n            Expected formats:\n                * https://www.googleapis.com/compute/v1/projects/[project]/zones/[zone]/disks/[disk_name]\n                * /projects/[project]/zones/[zone]/disks/[disk_name]\n                * /projects/[project]/regions/[region]/disks/[disk_name]\n        mode: Specifies in what mode the disk will be attached to the instance. Available options are `READ_ONLY`\n            and `READ_WRITE`. Disk in `READ_ONLY` mode can be attached to multiple instances at once.\n    '
    instances_client = compute_v1.InstancesClient()
    request = compute_v1.AttachDiskInstanceRequest()
    request.project = project_id
    request.zone = zone
    request.instance = instance_name
    request.attached_disk_resource = compute_v1.AttachedDisk()
    request.attached_disk_resource.source = disk_link
    request.attached_disk_resource.mode = mode
    operation = instances_client.attach_disk(request)
    wait_for_extended_operation(operation, 'disk attachement')