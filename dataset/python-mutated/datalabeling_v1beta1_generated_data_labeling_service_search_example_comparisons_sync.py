from google.cloud import datalabeling_v1beta1

def sample_search_example_comparisons():
    if False:
        print('Hello World!')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.SearchExampleComparisonsRequest(parent='parent_value')
    page_result = client.search_example_comparisons(request=request)
    for response in page_result:
        print(response)