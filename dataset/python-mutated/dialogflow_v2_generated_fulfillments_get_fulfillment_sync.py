from google.cloud import dialogflow_v2

def sample_get_fulfillment():
    if False:
        while True:
            i = 10
    client = dialogflow_v2.FulfillmentsClient()
    request = dialogflow_v2.GetFulfillmentRequest(name='name_value')
    response = client.get_fulfillment(request=request)
    print(response)