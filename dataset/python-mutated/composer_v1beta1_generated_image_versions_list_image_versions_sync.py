from google.cloud.orchestration.airflow import service_v1beta1

def sample_list_image_versions():
    if False:
        for i in range(10):
            print('nop')
    client = service_v1beta1.ImageVersionsClient()
    request = service_v1beta1.ListImageVersionsRequest()
    page_result = client.list_image_versions(request=request)
    for response in page_result:
        print(response)