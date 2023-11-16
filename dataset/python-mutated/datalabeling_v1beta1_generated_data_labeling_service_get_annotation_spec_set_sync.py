from google.cloud import datalabeling_v1beta1

def sample_get_annotation_spec_set():
    if False:
        print('Hello World!')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.GetAnnotationSpecSetRequest(name='name_value')
    response = client.get_annotation_spec_set(request=request)
    print(response)