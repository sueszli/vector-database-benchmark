"""
A sample script showing how to handle default values when communicating
with the Compute Engine API and how to configure usage reports using the API.
"""
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

def set_usage_export_bucket(project_id: str, bucket_name: str, report_name_prefix: str='') -> None:
    if False:
        i = 10
        return i + 15
    '\n    Set Compute Engine usage export bucket for the Cloud project.\n    This sample presents how to interpret the default value for the\n    report name prefix parameter.\n\n    Args:\n        project_id: project ID or project number of the project to update.\n        bucket_name: Google Cloud Storage bucket used to store Compute Engine\n            usage reports. An existing Google Cloud Storage bucket is required.\n        report_name_prefix: Prefix of the usage report name which defaults to an empty string\n            to showcase default values behaviour.\n    '
    usage_export_location = compute_v1.UsageExportLocation()
    usage_export_location.bucket_name = bucket_name
    usage_export_location.report_name_prefix = report_name_prefix
    if not report_name_prefix:
        print('Setting report_name_prefix to empty value causes the report to have the default prefix of `usage_gce`.')
    projects_client = compute_v1.ProjectsClient()
    operation = projects_client.set_usage_export_bucket(project=project_id, usage_export_location_resource=usage_export_location)
    wait_for_extended_operation(operation, 'setting GCE usage bucket')

def get_usage_export_bucket(project_id: str) -> compute_v1.UsageExportLocation:
    if False:
        print('Hello World!')
    '\n    Retrieve Compute Engine usage export bucket for the Cloud project.\n    Replaces the empty value returned by the API with the default value used\n    to generate report file names.\n\n    Args:\n        project_id: project ID or project number of the project to update.\n    Returns:\n        UsageExportLocation object describing the current usage export settings\n        for project project_id.\n    '
    projects_client = compute_v1.ProjectsClient()
    project_data = projects_client.get(project=project_id)
    uel = project_data.usage_export_location
    if not uel.bucket_name:
        return uel
    if not uel.report_name_prefix:
        print('Report name prefix not set, replacing with default value of `usage_gce`.')
        uel.report_name_prefix = 'usage_gce'
    return uel

def wait_for_extended_operation(operation: ExtendedOperation, verbose_name: str='operation', timeout: int=300) -> Any:
    if False:
        i = 10
        return i + 15
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

def disable_usage_export(project_id: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Disable Compute Engine usage export bucket for the Cloud Project.\n\n    Args:\n        project_id: project ID or project number of the project to update.\n    '
    projects_client = compute_v1.ProjectsClient()
    operation = projects_client.set_usage_export_bucket(project=project_id, usage_export_location_resource={})
    wait_for_extended_operation(operation, 'disabling GCE usage bucket')