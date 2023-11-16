from __future__ import annotations
import sys
from typing import Any
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

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

def create_instance_from_template_with_overrides(project_id: str, zone: str, instance_name: str, instance_template_name: str, machine_type: str, new_disk_source_image: str) -> compute_v1.Instance:
    if False:
        print('Hello World!')
    '\n    Creates a Compute Engine VM instance from an instance template, changing the machine type and\n    adding a new disk created from a source image.\n\n    Args:\n        project_id: ID or number of the project you want to use.\n        zone: Name of the zone you want to check, for example: us-west3-b\n        instance_name: Name of the new instance.\n        instance_template_name: Name of the instance template used for creating the new instance.\n        machine_type: Machine type you want to set in following format:\n            "zones/{zone}/machineTypes/{type_name}". For example:\n            - "zones/europe-west3-c/machineTypes/f1-micro"\n            - You can find the list of available machine types using:\n              https://cloud.google.com/sdk/gcloud/reference/compute/machine-types/list\n        new_disk_source_image: Path the the disk image you want to use for your new\n            disk. This can be one of the public images\n            (like "projects/debian-cloud/global/images/family/debian-10")\n            or a private image you have access to.\n            For a list of available public images, see the documentation:\n            http://cloud.google.com/compute/docs/images\n\n    Returns:\n        Instance object.\n    '
    instance_client = compute_v1.InstancesClient()
    instance_template_client = compute_v1.InstanceTemplatesClient()
    instance_template = instance_template_client.get(project=project_id, instance_template=instance_template_name)
    for disk in instance_template.properties.disks:
        if disk.initialize_params.disk_type:
            disk.initialize_params.disk_type = f'zones/{zone}/diskTypes/{disk.initialize_params.disk_type}'
    instance = compute_v1.Instance()
    instance.name = instance_name
    instance.machine_type = machine_type
    instance.disks = list(instance_template.properties.disks)
    new_disk = compute_v1.AttachedDisk()
    new_disk.initialize_params.disk_size_gb = 50
    new_disk.initialize_params.source_image = new_disk_source_image
    new_disk.auto_delete = True
    new_disk.boot = False
    new_disk.type_ = 'PERSISTENT'
    instance.disks.append(new_disk)
    instance_insert_request = compute_v1.InsertInstanceRequest()
    instance_insert_request.project = project_id
    instance_insert_request.zone = zone
    instance_insert_request.instance_resource = instance
    instance_insert_request.source_instance_template = instance_template.self_link
    operation = instance_client.insert(instance_insert_request)
    wait_for_extended_operation(operation, 'instance creation')
    return instance_client.get(project=project_id, zone=zone, instance=instance_name)