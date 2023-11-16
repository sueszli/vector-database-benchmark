from google.cloud import compute_v1

def get_instance_serial_port_output(project_id: str, zone: str, instance_name: str) -> compute_v1.SerialPortOutput:
    if False:
        return 10
    '\n    Returns the last 1 MB of serial port output from the specified instance.\n\n    Args:\n        project_id: project ID or project number of the Cloud project you want to use.\n        zone: name of the zone you want to use. For example: “us-west3-b”\n        instance_name: name of the VM instance you want to query.\n    Returns:\n        Content of the serial port output of an instance inside a compute_v1.SerialPortOutput object.\n        More about this type: https://cloud.google.com/python/docs/reference/compute/latest/google.cloud.compute_v1.types.SerialPortOutput\n\n    '
    instance_client = compute_v1.InstancesClient()
    return instance_client.get_serial_port_output(project=project_id, zone=zone, instance=instance_name)