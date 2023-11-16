from google.cloud import dialogflowcx_v3beta1

def sample_delete_webhook():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3beta1.WebhooksClient()
    request = dialogflowcx_v3beta1.DeleteWebhookRequest(name='name_value')
    client.delete_webhook(request=request)