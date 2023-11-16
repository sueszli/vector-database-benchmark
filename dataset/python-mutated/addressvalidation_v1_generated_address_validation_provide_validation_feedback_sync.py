from google.maps import addressvalidation_v1

def sample_provide_validation_feedback():
    if False:
        i = 10
        return i + 15
    client = addressvalidation_v1.AddressValidationClient()
    request = addressvalidation_v1.ProvideValidationFeedbackRequest(conclusion='UNUSED', response_id='response_id_value')
    response = client.provide_validation_feedback(request=request)
    print(response)