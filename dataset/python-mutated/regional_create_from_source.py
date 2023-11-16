from __future__ import annotations
from collections.abc import Iterable
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

def create_regional_disk(project_id: str, region: str, replica_zones: Iterable[str], disk_name: str, disk_type: str, disk_size_gb: int, disk_link: str | None=None, snapshot_link: str | None=None) -> compute_v1.Disk:
    if False:
        return 10
    '\n    Creates a regional disk from an existing zonal disk in a given project.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        region: name of the region in which you want to create the disk.\n        replica_zones: an iterable collection of zone names in which you want to keep\n            the new disks\' replicas. One of the replica zones of the clone must match\n            the zone of the source disk.\n        disk_name: name of the disk you want to create.\n        disk_type: the type of disk you want to create. This value uses the following format:\n            "regions/{region}/diskTypes/(pd-standard|pd-ssd|pd-balanced|pd-extreme)".\n            For example: "regions/us-west3/diskTypes/pd-ssd"\n        disk_size_gb: size of the new disk in gigabytes\n        disk_link: a link to the disk you want to use as a source for the new disk.\n            This value uses the following format: "projects/{project_name}/zones/{zone}/disks/{disk_name}"\n        snapshot_link: a link to the snapshot you want to use as a source for the new disk.\n            This value uses the following format: "projects/{project_name}/global/snapshots/{snapshot_name}"\n\n    Returns:\n        An attachable regional disk.\n    '
    disk_client = compute_v1.RegionDisksClient()
    disk = compute_v1.Disk()
    disk.replica_zones = replica_zones
    disk.size_gb = disk_size_gb
    if disk_link:
        disk.source_disk = disk_link
    if snapshot_link:
        disk.source_snapshot = snapshot_link
    disk.type_ = disk_type
    disk.region = region
    disk.name = disk_name
    operation = disk_client.insert(project=project_id, region=region, disk_resource=disk)
    wait_for_extended_operation(operation, 'disk creation')
    return disk_client.get(project=project_id, region=region, disk=disk_name)