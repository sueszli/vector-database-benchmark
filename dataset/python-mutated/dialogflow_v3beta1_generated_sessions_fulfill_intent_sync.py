from google.cloud import dialogflowcx_v3beta1

def sample_fulfill_intent():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.SessionsClient()
    request = dialogflowcx_v3beta1.FulfillIntentRequest()
    response = client.fulfill_intent(request=request)
    print(response)