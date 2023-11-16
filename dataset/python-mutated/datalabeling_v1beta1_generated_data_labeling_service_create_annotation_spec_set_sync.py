from google.cloud import datalabeling_v1beta1

def sample_create_annotation_spec_set():
    if False:
        return 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.CreateAnnotationSpecSetRequest(parent='parent_value')
    response = client.create_annotation_spec_set(request=request)
    print(response)