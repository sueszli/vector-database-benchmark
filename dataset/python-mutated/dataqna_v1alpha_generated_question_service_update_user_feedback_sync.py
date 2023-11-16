from google.cloud import dataqna_v1alpha

def sample_update_user_feedback():
    if False:
        while True:
            i = 10
    client = dataqna_v1alpha.QuestionServiceClient()
    user_feedback = dataqna_v1alpha.UserFeedback()
    user_feedback.name = 'name_value'
    request = dataqna_v1alpha.UpdateUserFeedbackRequest(user_feedback=user_feedback)
    response = client.update_user_feedback(request=request)
    print(response)