from google.cloud import dialogflow_v2beta1

def sample_create_intent():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2beta1.IntentsClient()
    intent = dialogflow_v2beta1.Intent()
    intent.display_name = 'display_name_value'
    request = dialogflow_v2beta1.CreateIntentRequest(parent='parent_value', intent=intent)
    response = client.create_intent(request=request)
    print(response)