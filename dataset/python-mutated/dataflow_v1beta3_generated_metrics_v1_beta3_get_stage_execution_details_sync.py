from google.cloud import dataflow_v1beta3

def sample_get_stage_execution_details():
    if False:
        i = 10
        return i + 15
    client = dataflow_v1beta3.MetricsV1Beta3Client()
    request = dataflow_v1beta3.GetStageExecutionDetailsRequest()
    page_result = client.get_stage_execution_details(request=request)
    for response in page_result:
        print(response)