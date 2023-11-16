def test_example_raw_response_hook():
    if False:
        print('Hello World!')

    def callback(response):
        if False:
            for i in range(10):
                print('nop')
        response.http_response.status_code = 200
        response.http_response.headers['custom_header'] = 'CustomHeader'
    from azure.core.pipeline import Pipeline
    from azure.core.rest import HttpRequest
    from azure.core.pipeline.policies import RedirectPolicy, UserAgentPolicy
    from azure.core.pipeline.transport import RequestsTransport
    from azure.core.pipeline.policies import CustomHookPolicy
    request = HttpRequest('GET', 'https://bing.com')
    policies = [CustomHookPolicy(raw_response_hook=callback)]
    with Pipeline(transport=RequestsTransport(), policies=policies) as pipeline:
        response = pipeline.run(request)
        assert response.http_response.status_code == 200
        assert response.http_response.headers['custom_header'] == 'CustomHeader'