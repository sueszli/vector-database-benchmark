from google.cloud import datalabeling_v1beta1

def sample_delete_dataset():
    if False:
        while True:
            i = 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.DeleteDatasetRequest(name='name_value')
    client.delete_dataset(request=request)