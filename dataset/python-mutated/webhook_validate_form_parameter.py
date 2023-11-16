""" DialogFlow CX: webhook to validate or invalidate form parameters snippet."""
import functions_framework

@functions_framework.http
def validate_parameter(request):
    if False:
        print('Hello World!')
    'Webhook will validate or invalidate parameter based on logic configured by the user.'
    return {'page_info': {'form_info': {'parameter_info': [{'displayName': 'orderNumber', 'required': True, 'state': 'INVALID', 'value': 123}]}}, 'sessionInfo': {'parameters': {'orderNumber': None}}}