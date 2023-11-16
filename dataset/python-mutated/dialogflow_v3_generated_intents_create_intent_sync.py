from google.cloud import dialogflowcx_v3

def sample_create_intent():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3.IntentsClient()
    intent = dialogflowcx_v3.Intent()
    intent.display_name = 'display_name_value'
    request = dialogflowcx_v3.CreateIntentRequest(parent='parent_value', intent=intent)
    response = client.create_intent(request=request)
    print(response)