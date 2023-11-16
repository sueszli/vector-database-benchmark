from google.cloud import datalabeling_v1beta1

def sample_create_dataset():
    if False:
        print('Hello World!')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.CreateDatasetRequest(parent='parent_value')
    response = client.create_dataset(request=request)
    print(response)