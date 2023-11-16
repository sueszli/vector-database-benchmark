from google.cloud import servicemanagement_v1

def sample_submit_config_source():
    if False:
        i = 10
        return i + 15
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.SubmitConfigSourceRequest(service_name='service_name_value')
    operation = client.submit_config_source(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)