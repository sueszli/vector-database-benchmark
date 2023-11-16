from google.cloud import datalabeling_v1beta1

def sample_list_examples():
    if False:
        while True:
            i = 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.ListExamplesRequest(parent='parent_value')
    page_result = client.list_examples(request=request)
    for response in page_result:
        print(response)