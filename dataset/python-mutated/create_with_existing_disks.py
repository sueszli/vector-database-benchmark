from __future__ import annotations
import re
import sys
from typing import Any
import warnings
from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1

def get_disk(project_id: str, zone: str, disk_name: str) -> compute_v1.Disk:
    if False:
        return 10
    '\n    Gets a disk from a project.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone where the disk exists.\n        disk_name: name of the disk you want to retrieve.\n    '
    disk_client = compute_v1.DisksClient()
    return disk_client.get(project=project_id, zone=zone, disk=disk_name)

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

def create_instance(project_id: str, zone: str, instance_name: str, disks: list[compute_v1.AttachedDisk], machine_type: str='n1-standard-1', network_link: str='global/networks/default', subnetwork_link: str=None, internal_ip: str=None, external_access: bool=False, external_ipv4: str=None, accelerators: list[compute_v1.AcceleratorConfig]=None, preemptible: bool=False, spot: bool=False, instance_termination_action: str='STOP', custom_hostname: str=None, delete_protection: bool=False) -> compute_v1.Instance:
    if False:
        for i in range(10):
            print('nop')
    '\n    Send an instance creation request to the Compute Engine API and wait for it to complete.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone to create the instance in. For example: "us-west3-b"\n        instance_name: name of the new virtual machine (VM) instance.\n        disks: a list of compute_v1.AttachedDisk objects describing the disks\n            you want to attach to your new instance.\n        machine_type: machine type of the VM being created. This value uses the\n            following format: "zones/{zone}/machineTypes/{type_name}".\n            For example: "zones/europe-west3-c/machineTypes/f1-micro"\n        network_link: name of the network you want the new instance to use.\n            For example: "global/networks/default" represents the network\n            named "default", which is created automatically for each project.\n        subnetwork_link: name of the subnetwork you want the new instance to use.\n            This value uses the following format:\n            "regions/{region}/subnetworks/{subnetwork_name}"\n        internal_ip: internal IP address you want to assign to the new instance.\n            By default, a free address from the pool of available internal IP addresses of\n            used subnet will be used.\n        external_access: boolean flag indicating if the instance should have an external IPv4\n            address assigned.\n        external_ipv4: external IPv4 address to be assigned to this instance. If you specify\n            an external IP address, it must live in the same region as the zone of the instance.\n            This setting requires `external_access` to be set to True to work.\n        accelerators: a list of AcceleratorConfig objects describing the accelerators that will\n            be attached to the new instance.\n        preemptible: boolean value indicating if the new instance should be preemptible\n            or not. Preemptible VMs have been deprecated and you should now use Spot VMs.\n        spot: boolean value indicating if the new instance should be a Spot VM or not.\n        instance_termination_action: What action should be taken once a Spot VM is terminated.\n            Possible values: "STOP", "DELETE"\n        custom_hostname: Custom hostname of the new VM instance.\n            Custom hostnames must conform to RFC 1035 requirements for valid hostnames.\n        delete_protection: boolean value indicating if the new virtual machine should be\n            protected against deletion or not.\n    Returns:\n        Instance object.\n    '
    instance_client = compute_v1.InstancesClient()
    network_interface = compute_v1.NetworkInterface()
    network_interface.network = network_link
    if subnetwork_link:
        network_interface.subnetwork = subnetwork_link
    if internal_ip:
        network_interface.network_i_p = internal_ip
    if external_access:
        access = compute_v1.AccessConfig()
        access.type_ = compute_v1.AccessConfig.Type.ONE_TO_ONE_NAT.name
        access.name = 'External NAT'
        access.network_tier = access.NetworkTier.PREMIUM.name
        if external_ipv4:
            access.nat_i_p = external_ipv4
        network_interface.access_configs = [access]
    instance = compute_v1.Instance()
    instance.network_interfaces = [network_interface]
    instance.name = instance_name
    instance.disks = disks
    if re.match('^zones/[a-z\\d\\-]+/machineTypes/[a-z\\d\\-]+$', machine_type):
        instance.machine_type = machine_type
    else:
        instance.machine_type = f'zones/{zone}/machineTypes/{machine_type}'
    instance.scheduling = compute_v1.Scheduling()
    if accelerators:
        instance.guest_accelerators = accelerators
        instance.scheduling.on_host_maintenance = compute_v1.Scheduling.OnHostMaintenance.TERMINATE.name
    if preemptible:
        warnings.warn('Preemptible VMs are being replaced by Spot VMs.', DeprecationWarning)
        instance.scheduling = compute_v1.Scheduling()
        instance.scheduling.preemptible = True
    if spot:
        instance.scheduling.provisioning_model = compute_v1.Scheduling.ProvisioningModel.SPOT.name
        instance.scheduling.instance_termination_action = instance_termination_action
    if custom_hostname is not None:
        instance.hostname = custom_hostname
    if delete_protection:
        instance.deletion_protection = True
    request = compute_v1.InsertInstanceRequest()
    request.zone = zone
    request.project = project_id
    request.instance_resource = instance
    print(f'Creating the {instance_name} instance in {zone}...')
    operation = instance_client.insert(request=request)
    wait_for_extended_operation(operation, 'instance creation')
    print(f'Instance {instance_name} created.')
    return instance_client.get(project=project_id, zone=zone, instance=instance_name)

def create_with_existing_disks(project_id: str, zone: str, instance_name: str, disk_names: list[str]) -> compute_v1.Instance:
    if False:
        i = 10
        return i + 15
    '\n    Create a new VM instance using selected disks. The first disk in disk_names will\n    be used as boot disk.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone to create the instance in. For example: "us-west3-b"\n        instance_name: name of the new virtual machine (VM) instance.\n        disk_names: list of disk names to be attached to the new virtual machine.\n            First disk in this list will be used as the boot device.\n\n    Returns:\n        Instance object.\n    '
    assert len(disk_names) >= 1
    disks = [get_disk(project_id, zone, disk_name) for disk_name in disk_names]
    attached_disks = []
    for disk in disks:
        adisk = compute_v1.AttachedDisk()
        adisk.source = disk.self_link
        attached_disks.append(adisk)
    attached_disks[0].boot = True
    instance = create_instance(project_id, zone, instance_name, attached_disks)
    return instance