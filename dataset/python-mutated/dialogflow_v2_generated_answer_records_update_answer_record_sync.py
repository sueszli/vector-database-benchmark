from google.cloud import dialogflow_v2

def sample_update_answer_record():
    if False:
        return 10
    client = dialogflow_v2.AnswerRecordsClient()
    request = dialogflow_v2.UpdateAnswerRecordRequest()
    response = client.update_answer_record(request=request)
    print(response)