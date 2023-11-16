from google.cloud import dialogflow_v2beta1

def sample_update_answer_record():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2beta1.AnswerRecordsClient()
    request = dialogflow_v2beta1.UpdateAnswerRecordRequest()
    response = client.update_answer_record(request=request)
    print(response)