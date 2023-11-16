from google.cloud import datalabeling_v1beta1

def sample_delete_annotated_dataset():
    if False:
        return 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.DeleteAnnotatedDatasetRequest(name='name_value')
    client.delete_annotated_dataset(request=request)