from google.cloud import webrisk_v1beta1

def sample_search_uris():
    if False:
        i = 10
        return i + 15
    client = webrisk_v1beta1.WebRiskServiceV1Beta1Client()
    request = webrisk_v1beta1.SearchUrisRequest(uri='uri_value', threat_types=['UNWANTED_SOFTWARE'])
    response = client.search_uris(request=request)
    print(response)