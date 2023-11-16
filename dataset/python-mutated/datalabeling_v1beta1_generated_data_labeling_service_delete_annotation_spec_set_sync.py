from google.cloud import datalabeling_v1beta1

def sample_delete_annotation_spec_set():
    if False:
        for i in range(10):
            print('nop')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.DeleteAnnotationSpecSetRequest(name='name_value')
    client.delete_annotation_spec_set(request=request)