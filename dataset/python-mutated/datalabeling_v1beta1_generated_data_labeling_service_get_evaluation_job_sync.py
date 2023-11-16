from google.cloud import datalabeling_v1beta1

def sample_get_evaluation_job():
    if False:
        for i in range(10):
            print('nop')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.GetEvaluationJobRequest(name='name_value')
    response = client.get_evaluation_job(request=request)
    print(response)