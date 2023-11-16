from google.cloud import dataflow_v1beta3

def sample_list_job_messages():
    if False:
        while True:
            i = 10
    client = dataflow_v1beta3.MessagesV1Beta3Client()
    request = dataflow_v1beta3.ListJobMessagesRequest()
    page_result = client.list_job_messages(request=request)
    for response in page_result:
        print(response)