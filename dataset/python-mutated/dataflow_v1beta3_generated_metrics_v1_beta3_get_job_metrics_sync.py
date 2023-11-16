from google.cloud import dataflow_v1beta3

def sample_get_job_metrics():
    if False:
        while True:
            i = 10
    client = dataflow_v1beta3.MetricsV1Beta3Client()
    request = dataflow_v1beta3.GetJobMetricsRequest()
    response = client.get_job_metrics(request=request)
    print(response)