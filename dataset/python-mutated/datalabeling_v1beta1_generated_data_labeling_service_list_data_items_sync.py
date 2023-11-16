from google.cloud import datalabeling_v1beta1

def sample_list_data_items():
    if False:
        i = 10
        return i + 15
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.ListDataItemsRequest(parent='parent_value')
    page_result = client.list_data_items(request=request)
    for response in page_result:
        print(response)