from google.cloud import dataqna_v1alpha

def sample_get_user_feedback():
    if False:
        for i in range(10):
            print('nop')
    client = dataqna_v1alpha.QuestionServiceClient()
    request = dataqna_v1alpha.GetUserFeedbackRequest(name='name_value')
    response = client.get_user_feedback(request=request)
    print(response)