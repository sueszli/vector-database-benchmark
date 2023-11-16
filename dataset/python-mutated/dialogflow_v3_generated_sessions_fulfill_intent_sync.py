from google.cloud import dialogflowcx_v3

def sample_fulfill_intent():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3.SessionsClient()
    request = dialogflowcx_v3.FulfillIntentRequest()
    response = client.fulfill_intent(request=request)
    print(response)