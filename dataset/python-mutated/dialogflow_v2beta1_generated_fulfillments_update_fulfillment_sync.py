from google.cloud import dialogflow_v2beta1

def sample_update_fulfillment():
    if False:
        print('Hello World!')
    client = dialogflow_v2beta1.FulfillmentsClient()
    fulfillment = dialogflow_v2beta1.Fulfillment()
    fulfillment.generic_web_service.uri = 'uri_value'
    fulfillment.name = 'name_value'
    request = dialogflow_v2beta1.UpdateFulfillmentRequest(fulfillment=fulfillment)
    response = client.update_fulfillment(request=request)
    print(response)