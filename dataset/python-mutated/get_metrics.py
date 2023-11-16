from google.cloud import recaptchaenterprise_v1

def get_metrics(project_id: str, recaptcha_site_key: str) -> None:
    if False:
        return 10
    'Get metrics specific to a recaptcha site key.\n        E.g: score bucket count for a key or number of\n        times the checkbox key failed/ passed etc.,\n    Args:\n    project_id: Google Cloud Project ID.\n    recaptcha_site_key: Specify the site key to get metrics.\n    '
    client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
    metrics_name = f'projects/{project_id}/keys/{recaptcha_site_key}/metrics'
    request = recaptchaenterprise_v1.GetMetricsRequest()
    request.name = metrics_name
    response = client.get_metrics(request)
    for day_metric in response.score_metrics:
        score_bucket_count = day_metric.overall_metrics.score_buckets
        print(score_bucket_count)
    print(f'Retrieved the bucket count for score based key: {recaptcha_site_key}')