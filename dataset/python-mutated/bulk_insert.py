from __future__ import annotations
from collections.abc import Iterable
import sys
from typing import Any
import uuid
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

def get_instance_template(project_id: str, template_name: str) -> compute_v1.InstanceTemplate:
    if False:
        i = 10
        return i + 15
    '\n    Retrieve an instance template, which you can use to create virtual machine\n    (VM) instances and managed instance groups (MIGs).\n\n    Args:\n        project_id: project ID or project number of the Cloud project you use.\n        template_name: name of the template to retrieve.\n\n    Returns:\n        InstanceTemplate object that represents the retrieved template.\n    '
    template_client = compute_v1.InstanceTemplatesClient()
    return template_client.get(project=project_id, instance_template=template_name)

def bulk_insert_instance(project_id: str, zone: str, template: compute_v1.InstanceTemplate, count: int, name_pattern: str, min_count: int | None=None, labels: dict | None=None) -> Iterable[compute_v1.Instance]:
    if False:
        print('Hello World!')
    '\n    Create multiple VMs based on an Instance Template. The newly created instances will\n    be returned as a list and will share a label with key `bulk_batch` and a random\n    value.\n\n    If the bulk insert operation fails and the requested number of instances can\'t be created,\n    and more than min_count instances are created, then those instances can be found using\n    the `bulk_batch` label with value attached to the raised exception in bulk_batch_id\n    attribute. So, you can use the following filter: f"label.bulk_batch={err.bulk_batch_id}"\n    when listing instances in a zone to get the instances that were successfully created.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone to create the instance in. For example: "us-west3-b"\n        template: an Instance Template to be used for creation of the new VMs.\n        name_pattern: The string pattern used for the names of the VMs. The pattern\n            must contain one continuous sequence of placeholder hash characters (#)\n            with each character corresponding to one digit of the generated instance\n            name. Example: a name_pattern of inst-#### generates instance names such\n            as inst-0001 and inst-0002. If existing instances in the same project and\n            zone have names that match the name pattern then the generated instance\n            numbers start after the biggest existing number. For example, if there\n            exists an instance with name inst-0050, then instance names generated\n            using the pattern inst-#### begin with inst-0051. The name pattern\n            placeholder #...# can contain up to 18 characters.\n        count: The maximum number of instances to create.\n        min_count (optional): The minimum number of instances to create. If no min_count is\n            specified then count is used as the default value. If min_count instances\n            cannot be created, then no instances will be created and instances already\n            created will be deleted.\n        labels (optional): A dictionary with labels to be added to the new VMs.\n    '
    bulk_insert_resource = compute_v1.BulkInsertInstanceResource()
    bulk_insert_resource.source_instance_template = template.self_link
    bulk_insert_resource.count = count
    bulk_insert_resource.min_count = min_count or count
    bulk_insert_resource.name_pattern = name_pattern
    if not labels:
        labels = {}
    labels['bulk_batch'] = uuid.uuid4().hex
    instance_prop = compute_v1.InstanceProperties()
    instance_prop.labels = labels
    bulk_insert_resource.instance_properties = instance_prop
    bulk_insert_request = compute_v1.BulkInsertInstanceRequest()
    bulk_insert_request.bulk_insert_instance_resource_resource = bulk_insert_resource
    bulk_insert_request.project = project_id
    bulk_insert_request.zone = zone
    client = compute_v1.InstancesClient()
    operation = client.bulk_insert(bulk_insert_request)
    try:
        wait_for_extended_operation(operation, 'bulk instance creation')
    except Exception as err:
        err.bulk_batch_id = labels['bulk_batch']
        raise err
    list_req = compute_v1.ListInstancesRequest()
    list_req.project = project_id
    list_req.zone = zone
    list_req.filter = ' AND '.join((f'labels.{key}:{value}' for (key, value) in labels.items()))
    return client.list(list_req)

def create_five_instances(project_id: str, zone: str, template_name: str, name_pattern: str):
    if False:
        while True:
            i = 10
    '\n    Create five instances of an instance template.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone to create the instance in. For example: "us-west3-b"\n        template_name: name of the template that will be used to create new VMs.\n        name_pattern: The string pattern used for the names of the VMs.\n    '
    template = get_instance_template(project_id, template_name)
    instances = bulk_insert_instance(project_id, zone, template, 5, name_pattern)
    return instances