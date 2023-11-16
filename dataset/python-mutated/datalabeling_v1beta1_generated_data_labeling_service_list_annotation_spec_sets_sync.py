from google.cloud import datalabeling_v1beta1

def sample_list_annotation_spec_sets():
    if False:
        i = 10
        return i + 15
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.ListAnnotationSpecSetsRequest(parent='parent_value')
    page_result = client.list_annotation_spec_sets(request=request)
    for response in page_result:
        print(response)