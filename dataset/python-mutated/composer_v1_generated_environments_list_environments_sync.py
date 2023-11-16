from google.cloud.orchestration.airflow import service_v1

def sample_list_environments():
    if False:
        while True:
            i = 10
    client = service_v1.EnvironmentsClient()
    request = service_v1.ListEnvironmentsRequest()
    page_result = client.list_environments(request=request)
    for response in page_result:
        print(response)