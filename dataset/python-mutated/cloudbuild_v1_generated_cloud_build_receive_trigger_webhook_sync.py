from google.cloud.devtools import cloudbuild_v1

def sample_receive_trigger_webhook():
    if False:
        return 10
    client = cloudbuild_v1.CloudBuildClient()
    request = cloudbuild_v1.ReceiveTriggerWebhookRequest()
    response = client.receive_trigger_webhook(request=request)
    print(response)