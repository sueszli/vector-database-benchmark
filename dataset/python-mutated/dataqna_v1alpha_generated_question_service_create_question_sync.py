from google.cloud import dataqna_v1alpha

def sample_create_question():
    if False:
        i = 10
        return i + 15
    client = dataqna_v1alpha.QuestionServiceClient()
    question = dataqna_v1alpha.Question()
    question.scopes = ['scopes_value1', 'scopes_value2']
    question.query = 'query_value'
    request = dataqna_v1alpha.CreateQuestionRequest(parent='parent_value', question=question)
    response = client.create_question(request=request)
    print(response)