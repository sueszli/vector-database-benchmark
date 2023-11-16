from google.cloud import dialogflow_v2beta1

def sample_get_answer_record():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.AnswerRecordsClient()
    request = dialogflow_v2beta1.GetAnswerRecordRequest()
    response = client.get_answer_record(request=request)
    print(response)