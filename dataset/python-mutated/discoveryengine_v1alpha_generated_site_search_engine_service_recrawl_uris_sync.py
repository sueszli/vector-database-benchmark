from google.cloud import discoveryengine_v1alpha

def sample_recrawl_uris():
    if False:
        while True:
            i = 10
    client = discoveryengine_v1alpha.SiteSearchEngineServiceClient()
    request = discoveryengine_v1alpha.RecrawlUrisRequest(site_search_engine='site_search_engine_value', uris=['uris_value1', 'uris_value2'])
    operation = client.recrawl_uris(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)