from __future__ import annotations
import sys
from typing import Any
import warnings
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
STOPPED_MACHINE_STATUS = (compute_v1.Instance.Status.TERMINATED.name, compute_v1.Instance.Status.STOPPED.name)

def create_image_from_disk(project_id: str, zone: str, source_disk_name: str, image_name: str, storage_location: str | None=None, force_create: bool=False) -> compute_v1.Image:
    if False:
        i = 10
        return i + 15
    "\n    Creates a new disk image.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you use.\n        zone: zone of the disk you copy from.\n        source_disk_name: name of the source disk you copy from.\n        image_name: name of the image you want to create.\n        storage_location: storage location for the image. If the value is undefined,\n            function will store the image in the multi-region closest to your image's\n            source location.\n        force_create: create the image even if the source disk is attached to a\n            running instance.\n\n    Returns:\n        An Image object.\n    "
    image_client = compute_v1.ImagesClient()
    disk_client = compute_v1.DisksClient()
    instance_client = compute_v1.InstancesClient()
    disk = disk_client.get(project=project_id, zone=zone, disk=source_disk_name)
    for disk_user in disk.users:
        instance = instance_client.get(project=project_id, zone=zone, instance=disk_user)
        if instance.status in STOPPED_MACHINE_STATUS:
            continue
        if not force_create:
            raise RuntimeError(f'Instance {disk_user} should be stopped. For Windows instances please stop the instance using `GCESysprep` command. For Linux instances just shut it down normally. You can supress this error and create an image ofthe disk by setting `force_create` parameter to true (not recommended). \nMore information here: \n * https://cloud.google.com/compute/docs/instances/windows/creating-windows-os-image#api \n * https://cloud.google.com/compute/docs/images/create-delete-deprecate-private-images#prepare_instance_for_image')
        else:
            warnings.warn(f'Warning: The `force_create` option may compromise the integrity of your image. Stop the {disk_user} instance before you create the image if possible.')
    image = compute_v1.Image()
    image.source_disk = disk.self_link
    image.name = image_name
    if storage_location:
        image.storage_locations = [storage_location]
    operation = image_client.insert(project=project_id, image_resource=image)
    wait_for_extended_operation(operation, 'image creation from disk')
    return image_client.get(project=project_id, image=image_name)