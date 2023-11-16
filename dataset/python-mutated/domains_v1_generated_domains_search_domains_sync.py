from google.cloud import domains_v1

def sample_search_domains():
    if False:
        for i in range(10):
            print('nop')
    client = domains_v1.DomainsClient()
    request = domains_v1.SearchDomainsRequest(query='query_value', location='location_value')
    response = client.search_domains(request=request)
    print(response)