from google.cloud import datalabeling_v1beta1

def sample_list_evaluation_jobs():
    if False:
        for i in range(10):
            print('nop')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.ListEvaluationJobsRequest(parent='parent_value')
    page_result = client.list_evaluation_jobs(request=request)
    for response in page_result:
        print(response)