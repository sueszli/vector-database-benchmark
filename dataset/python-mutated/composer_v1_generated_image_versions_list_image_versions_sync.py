from google.cloud.orchestration.airflow import service_v1

def sample_list_image_versions():
    if False:
        while True:
            i = 10
    client = service_v1.ImageVersionsClient()
    request = service_v1.ListImageVersionsRequest()
    page_result = client.list_image_versions(request=request)
    for response in page_result:
        print(response)