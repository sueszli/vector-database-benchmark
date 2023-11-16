from google.cloud import servicemanagement_v1

def sample_get_service_rollout():
    if False:
        return 10
    client = servicemanagement_v1.ServiceManagerClient()
    request = servicemanagement_v1.GetServiceRolloutRequest(service_name='service_name_value', rollout_id='rollout_id_value')
    response = client.get_service_rollout(request=request)
    print(response)