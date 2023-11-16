from google.cloud import compute_v1

def sample_get_serial_port_output():
    if False:
        print('Hello World!')
    client = compute_v1.InstancesClient()
    request = compute_v1.GetSerialPortOutputInstanceRequest(instance='instance_value', project='project_value', zone='zone_value')
    response = client.get_serial_port_output(request=request)
    print(response)