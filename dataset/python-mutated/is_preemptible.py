from google.cloud import compute_v1

def is_preemptible(project_id: str, zone: str, instance_name: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if a given instance is preemptible or not.\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone you want to use. For example: "us-west3-b"\n        instance_name: name of the virtual machine to check.\n    Returns:\n        The preemptible status of the instance.\n    '
    instance_client = compute_v1.InstancesClient()
    instance = instance_client.get(project=project_id, zone=zone, instance=instance_name)
    return instance.scheduling.preemptible