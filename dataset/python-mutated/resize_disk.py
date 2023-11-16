from __future__ import annotations
import re
import sys
from typing import Any
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

def wait_for_extended_operation(operation: ExtendedOperation, verbose_name: str='operation', timeout: int=300) -> Any:
    if False:
        return 10
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

def resize_disk(project_id: str, disk_link: str, new_size_gb: int) -> None:
    if False:
        return 10
    '\n    Resizes a persistent disk to a specified size in GB. After you resize the disk, you must\n    also resize the file system so that the operating system can access the additional space.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        disk_link: a link to the disk you want to resize.\n            This value uses the following format:\n                * https://www.googleapis.com/compute/v1/projects/{project_name}/zones/{zone}/disks/{disk_name}\n                * projects/{project_name}/zones/{zone}/disks/{disk_name}\n                * projects/{project_name}/regions/{region}/disks/{disk_name}\n        new_size_gb: the new size you want to set for the disk in gigabytes.\n    '
    search_results = re.search('/projects/[\\w_-]+/(?P<area_type>zones|regions)/(?P<area_name>[\\w_-]+)/disks/(?P<disk_name>[\\w_-]+)', disk_link)
    if search_results['area_type'] == 'regions':
        disk_client = compute_v1.RegionDisksClient()
        request = compute_v1.ResizeRegionDiskRequest()
        request.region = search_results['area_name']
        request.region_disks_resize_request_resource = compute_v1.RegionDisksResizeRequest()
        request.region_disks_resize_request_resource.size_gb = new_size_gb
    else:
        disk_client = compute_v1.DisksClient()
        request = compute_v1.ResizeDiskRequest()
        request.zone = search_results['area_name']
        request.disks_resize_request_resource = compute_v1.DisksResizeRequest()
        request.disks_resize_request_resource.size_gb = new_size_gb
    request.disk = search_results['disk_name']
    request.project = project_id
    operation = disk_client.resize(request)
    wait_for_extended_operation(operation, 'disk resize')