from google.cloud import dataqna_v1alpha

def sample_get_question():
    if False:
        i = 10
        return i + 15
    client = dataqna_v1alpha.QuestionServiceClient()
    request = dataqna_v1alpha.GetQuestionRequest(name='name_value')
    response = client.get_question(request=request)
    print(response)