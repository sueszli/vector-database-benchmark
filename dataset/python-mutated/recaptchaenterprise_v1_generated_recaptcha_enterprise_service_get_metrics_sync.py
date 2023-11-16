from google.cloud import recaptchaenterprise_v1

def sample_get_metrics():
    if False:
        return 10
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    request = recaptchaenterprise_v1.GetMetricsRequest(name='name_value')
    response = client.get_metrics(request=request)
    print(response)