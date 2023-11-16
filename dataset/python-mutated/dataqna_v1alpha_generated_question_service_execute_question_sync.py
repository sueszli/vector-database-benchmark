from google.cloud import dataqna_v1alpha

def sample_execute_question():
    if False:
        return 10
    client = dataqna_v1alpha.QuestionServiceClient()
    request = dataqna_v1alpha.ExecuteQuestionRequest(name='name_value', interpretation_index=2159)
    response = client.execute_question(request=request)
    print(response)