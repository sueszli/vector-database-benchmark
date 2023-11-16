from google.cloud import datalabeling_v1beta1

def sample_update_evaluation_job():
    if False:
        i = 10
        return i + 15
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.UpdateEvaluationJobRequest()
    response = client.update_evaluation_job(request=request)
    print(response)