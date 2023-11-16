import google.auth
from google.cloud.retail import SearchRequest, SearchServiceClient
project_id = google.auth.default()[1]

def get_search_request(query: str, facet_key_param: str):
    if False:
        while True:
            i = 10
    default_search_placement = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/placements/default_search'
    facet_key = SearchRequest.FacetSpec().FacetKey()
    facet_key.key = facet_key_param
    facet_spec = SearchRequest.FacetSpec()
    facet_spec.facet_key = facet_key
    search_request = SearchRequest()
    search_request.placement = default_search_placement
    search_request.query = query
    search_request.visitor_id = '123456'
    search_request.facet_specs = [facet_spec]
    search_request.page_size = 10
    print('---search request---')
    print(search_request)
    return search_request

def search():
    if False:
        for i in range(10):
            print('nop')
    facet_key = 'colorFamilies'
    search_request = get_search_request('Tee', facet_key)
    search_response = SearchServiceClient().search(search_request)
    print('---search response---')
    if not search_response.results:
        print('The search operation returned no matching results.')
    else:
        print(search_response)
        print('---facets:---')
        print(search_response.facets)
    return search_response
search()