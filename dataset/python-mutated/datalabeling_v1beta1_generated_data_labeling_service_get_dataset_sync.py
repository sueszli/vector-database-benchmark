from google.cloud import datalabeling_v1beta1

def sample_get_dataset():
    if False:
        return 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.GetDatasetRequest(name='name_value')
    response = client.get_dataset(request=request)
    print(response)