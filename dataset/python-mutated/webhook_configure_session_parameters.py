""" DialogFlow CX: webhook to configure new session parameters."""
import functions_framework

@functions_framework.http
def configure_session_params(request):
    if False:
        print('Hello World!')
    'Webhook to validate or configure new session parameters.'
    order_number = 123
    json_response = {'sessionInfo': {'parameters': {'orderNumber': order_number}}}
    return json_response