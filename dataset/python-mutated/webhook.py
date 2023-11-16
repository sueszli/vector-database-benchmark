""" handle_webhook will return the correct fullfilment response dependong the tag that is sent in the request"""
import functions_framework

@functions_framework.http
def handle_webhook(request):
    if False:
        while True:
            i = 10
    req = request.get_json()
    tag = req['fulfillmentInfo']['tag']
    if tag == 'Default Welcome Intent':
        text = 'Hello from a GCF Webhook'
    elif tag == 'get-name':
        text = 'My name is Flowhook'
    else:
        text = f'There are no fulfillment responses defined for {tag} tag'
    res = {'fulfillment_response': {'messages': [{'text': {'text': [text]}}]}}
    return res