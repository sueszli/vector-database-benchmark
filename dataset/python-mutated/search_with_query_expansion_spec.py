import google.auth
from google.cloud.retail import SearchRequest, SearchServiceClient
project_id = google.auth.default()[1]

def get_search_request(query: str, condition: SearchRequest.QueryExpansionSpec.Condition):
    if False:
        while True:
            i = 10
    default_search_placement = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/placements/default_search'
    query_expansion_spec = SearchRequest().QueryExpansionSpec()
    query_expansion_spec.condition = condition
    search_request = SearchRequest()
    search_request.placement = default_search_placement
    search_request.query = query
    search_request.visitor_id = '123456'
    search_request.query_expansion_spec = query_expansion_spec
    search_request.page_size = 10
    print('---search request:---')
    print(search_request)
    return search_request

def search():
    if False:
        i = 10
        return i + 15
    condition = SearchRequest.QueryExpansionSpec.Condition.AUTO
    search_request = get_search_request('Google Youth Hero Tee Grey', condition)
    search_response = SearchServiceClient().search(search_request)
    print('---search response---')
    if not search_response.results:
        print('The search operation returned no matching results.')
    else:
        print(search_response)
    return search_response
search()