from google.cloud import datalabeling_v1beta1

def sample_create_evaluation_job():
    if False:
        for i in range(10):
            print('nop')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.CreateEvaluationJobRequest(parent='parent_value')
    response = client.create_evaluation_job(request=request)
    print(response)