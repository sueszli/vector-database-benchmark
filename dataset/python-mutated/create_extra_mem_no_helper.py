from __future__ import annotations
from google.cloud import compute_v1

def create_custom_instances_extra_mem(project_id: str, zone: str, instance_name: str, core_count: int, memory: int) -> list[compute_v1.Instance]:
    if False:
        while True:
            i = 10
    '\n    Create 3 new VM instances with extra memory without using a CustomMachineType helper class.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone to create the instance in. For example: "us-west3-b"\n        instance_name: name of the new virtual machine (VM) instance.\n        core_count: number of CPU cores you want to use.\n        memory: the amount of memory for the VM instance, in megabytes.\n\n    Returns:\n        List of Instance objects.\n    '
    newest_debian = get_image_from_family(project='debian-cloud', family='debian-10')
    disk_type = f'zones/{zone}/diskTypes/pd-standard'
    disks = [disk_from_image(disk_type, 10, True, newest_debian.self_link)]
    instances = [create_instance(project_id, zone, f'{instance_name}_n1_extra_mem', disks, f'zones/{zone}/machineTypes/custom-{core_count}-{memory}-ext'), create_instance(project_id, zone, f'{instance_name}_n2_extra_mem', disks, f'zones/{zone}/machineTypes/n2-custom-{core_count}-{memory}-ext'), create_instance(project_id, zone, f'{instance_name}_n2d_extra_mem', disks, f'zones/{zone}/machineTypes/n2d-custom-{core_count}-{memory}-ext')]
    return instances