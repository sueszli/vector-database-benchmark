from google.cloud import dialogflow_v2

def sample_list_answer_records():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.AnswerRecordsClient()
    request = dialogflow_v2.ListAnswerRecordsRequest(parent='parent_value')
    page_result = client.list_answer_records(request=request)
    for response in page_result:
        print(response)