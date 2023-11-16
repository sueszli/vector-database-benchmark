from google.cloud import datalabeling_v1beta1

def sample_list_annotated_datasets():
    if False:
        print('Hello World!')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.ListAnnotatedDatasetsRequest(parent='parent_value')
    page_result = client.list_annotated_datasets(request=request)
    for response in page_result:
        print(response)