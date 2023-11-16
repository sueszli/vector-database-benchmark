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

def create_template_from_instance(project_id: str, instance: str, template_name: str) -> compute_v1.InstanceTemplate:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a new instance template based on an existing instance.\n    This new template specifies a different boot disk.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you use.\n        instance: the instance to base the new template on. This value uses\n            the following format: "projects/{project}/zones/{zone}/instances/{instance_name}"\n        template_name: name of the new template to create.\n\n    Returns:\n        InstanceTemplate object that represents the new instance template.\n    '
    disk = compute_v1.DiskInstantiationConfig()
    disk.device_name = 'disk-1'
    disk.instantiate_from = 'CUSTOM_IMAGE'
    disk.custom_image = 'projects/rocky-linux-cloud/global/images/family/rocky-linux-8'
    disk.auto_delete = True
    template = compute_v1.InstanceTemplate()
    template.name = template_name
    template.source_instance = instance
    template.source_instance_params = compute_v1.SourceInstanceParams()
    template.source_instance_params.disk_configs = [disk]
    template_client = compute_v1.InstanceTemplatesClient()
    operation = template_client.insert(project=project_id, instance_template_resource=template)
    wait_for_extended_operation(operation, 'instance template creation')
    return template_client.get(project=project_id, instance_template=template_name)