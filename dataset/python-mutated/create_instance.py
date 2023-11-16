from __future__ import annotations
import re
import warnings
from google.cloud import compute_v1

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