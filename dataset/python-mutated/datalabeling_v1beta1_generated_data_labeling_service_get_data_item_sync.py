from google.cloud import datalabeling_v1beta1

def sample_get_data_item():
    if False:
        while True:
            i = 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.GetDataItemRequest(name='name_value')
    response = client.get_data_item(request=request)
    print(response)