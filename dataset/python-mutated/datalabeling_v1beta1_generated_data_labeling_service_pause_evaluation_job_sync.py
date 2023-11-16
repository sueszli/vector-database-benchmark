from google.cloud import datalabeling_v1beta1

def sample_pause_evaluation_job():
    if False:
        print('Hello World!')
    client = datalabeling_v1beta1.DataLabelingServiceClient()
    request = datalabeling_v1beta1.PauseEvaluationJobRequest(name='name_value')
    client.pause_evaluation_job(request=request)