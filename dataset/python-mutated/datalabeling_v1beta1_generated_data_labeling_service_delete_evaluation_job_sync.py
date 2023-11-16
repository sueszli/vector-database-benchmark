from google.cloud import datalabeling_v1beta1

def sample_delete_evaluation_job():
    if False:
        while True:
            i = 10
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.DeleteEvaluationJobRequest(name='name_value')
    client.delete_evaluation_job(request=request)