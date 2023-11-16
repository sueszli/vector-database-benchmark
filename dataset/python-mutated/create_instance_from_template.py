from google.cloud import compute_v1

def create_instance_from_template(project_id: str, zone: str, instance_name: str, instance_template_url: str) -> compute_v1.Instance:
    if False:
        return 10
    '\n    Creates a Compute Engine VM instance from an instance template.\n\n    Args:\n        project_id: ID or number of the project you want to use.\n        zone: Name of the zone you want to check, for example: us-west3-b\n        instance_name: Name of the new instance.\n        instance_template_url: URL of the instance template used for creating the new instance.\n            It can be a full or partial URL.\n            Examples:\n            - https://www.googleapis.com/compute/v1/projects/project/global/instanceTemplates/example-instance-template\n            - projects/project/global/instanceTemplates/example-instance-template\n            - global/instanceTemplates/example-instance-template\n\n    Returns:\n        Instance object.\n    '
    instance_client = compute_v1.InstancesClient()
    instance_insert_request = compute_v1.InsertInstanceRequest()
    instance_insert_request.project = project_id
    instance_insert_request.zone = zone
    instance_insert_request.source_instance_template = instance_template_url
    instance_insert_request.instance_resource.name = instance_name
    operation = instance_client.insert(instance_insert_request)
    wait_for_extended_operation(operation, 'instance creation')
    return instance_client.get(project=project_id, zone=zone, instance=instance_name)