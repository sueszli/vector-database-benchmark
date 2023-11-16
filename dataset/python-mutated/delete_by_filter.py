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

def delete_snapshot(project_id: str, snapshot_name: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Delete a snapshot of a disk.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        snapshot_name: name of the snapshot to delete.\n    '
    snapshot_client = compute_v1.SnapshotsClient()
    operation = snapshot_client.delete(project=project_id, snapshot=snapshot_name)
    wait_for_extended_operation(operation, 'snapshot deletion')

def list_snapshots(project_id: str, filter_: str='') -> Iterable[compute_v1.Snapshot]:
    if False:
        print('Hello World!')
    '\n    List snapshots from a project.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        filter_: filter to be applied when listing snapshots. Learn more about filters here:\n            https://cloud.google.com/python/docs/reference/compute/latest/google.cloud.compute_v1.types.ListSnapshotsRequest\n\n    Returns:\n        An iterable containing all Snapshots that match the provided filter.\n    '
    snapshot_client = compute_v1.SnapshotsClient()
    request = compute_v1.ListSnapshotsRequest()
    request.project = project_id
    request.filter = filter_
    return snapshot_client.list(request)

def delete_snapshots_by_filter(project_id: str, filter: str):
    if False:
        print('Hello World!')
    '\n    Deletes all snapshots in project that meet the filter criteria.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        filter: filter to be applied when looking for snapshots for deletion.\n    '
    for snapshot in list_snapshots(project_id, filter):
        delete_snapshot(project_id, snapshot.name)