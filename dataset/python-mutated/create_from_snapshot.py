from __future__ import annotations
from collections.abc import Iterable
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

def create_image_from_snapshot(project_id: str, source_snapshot_name: str, image_name: str, source_project_id: str | None=None, guest_os_features: Iterable[str] | None=None, storage_location: str | None=None) -> compute_v1.Image:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates an image based on a snapshot.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to place your new image in.\n        source_snapshot_name: name of the snapshot you want to use as a base of your image.\n        image_name: name of the image you want to create.\n        source_project_id: name of the project that hosts the source snapshot. If left unset, it\'s assumed to equal\n            the `project_id`.\n        guest_os_features: an iterable collection of guest features you want to enable for the bootable image.\n            Learn more about Guest OS features here:\n            https://cloud.google.com/compute/docs/images/create-delete-deprecate-private-images#guest-os-features\n        storage_location: the storage location of your image. For example, specify "us" to store the image in the\n            `us` multi-region, or "us-central1" to store it in the `us-central1` region. If you do not make a selection,\n             Compute Engine stores the image in the multi-region closest to your image\'s source location.\n\n    Returns:\n        An Image object.\n    '
    if source_project_id is None:
        source_project_id = project_id
    snapshot_client = compute_v1.SnapshotsClient()
    image_client = compute_v1.ImagesClient()
    src_snapshot = snapshot_client.get(project=source_project_id, snapshot=source_snapshot_name)
    image = compute_v1.Image()
    image.name = image_name
    image.source_snapshot = src_snapshot.self_link
    if storage_location:
        image.storage_locations = [storage_location]
    if guest_os_features:
        image.guest_os_features = [compute_v1.GuestOsFeature(type_=feature) for feature in guest_os_features]
    operation = image_client.insert(project=project_id, image_resource=image)
    wait_for_extended_operation(operation, 'image creation from snapshot')
    return image_client.get(project=project_id, image=image_name)