from google.cloud import run_v2

def sample_get_service():
    if False:
        print('Hello World!')
    client = run_v2.ServicesClient()
    request = run_v2.GetServiceRequest(name='name_value')
    response = client.get_service(request=request)
    print(response)