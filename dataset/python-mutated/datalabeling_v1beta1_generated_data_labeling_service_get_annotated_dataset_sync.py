from google.cloud import datalabeling_v1beta1

def sample_get_annotated_dataset():
    if False:
        while True:
            i = 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.GetAnnotatedDatasetRequest(name='name_value')
    response = client.get_annotated_dataset(request=request)
    print(response)