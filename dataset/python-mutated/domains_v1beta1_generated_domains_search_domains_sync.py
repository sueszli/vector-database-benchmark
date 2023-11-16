from google.cloud import domains_v1beta1

def sample_search_domains():
    if False:
        i = 10
        return i + 15
    client = domains_v1beta1.DomainsClient()
    request = domains_v1beta1.SearchDomainsRequest(query='query_value', location='location_value')
    response = client.search_domains(request=request)
    print(response)