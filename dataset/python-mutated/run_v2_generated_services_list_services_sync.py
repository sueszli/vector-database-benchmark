from google.cloud import run_v2

def sample_list_services():
    if False:
        while True:
            i = 10
    client = run_v2.ServicesClient()
    request = run_v2.ListServicesRequest(parent='parent_value')
    page_result = client.list_services(request=request)
    for response in page_result:
        print(response)