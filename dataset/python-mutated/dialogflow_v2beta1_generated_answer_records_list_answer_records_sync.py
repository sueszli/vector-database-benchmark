from google.cloud import dialogflow_v2beta1

def sample_list_answer_records():
    if False:
        while True:
            i = 10
    client = dialogflow_v2beta1.AnswerRecordsClient()
    request = dialogflow_v2beta1.ListAnswerRecordsRequest()
    page_result = client.list_answer_records(request=request)
    for response in page_result:
        print(response)