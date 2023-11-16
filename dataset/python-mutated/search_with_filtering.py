import google.auth
from google.cloud.retail import SearchRequest, SearchServiceClient
project_id = google.auth.default()[1]

def get_search_request(query: str, _filter: str):
    if False:
        return 10
    default_search_placement = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/placements/default_search'
    search_request = SearchRequest()
    search_request.placement = default_search_placement
    search_request.query = query
    search_request.filter = _filter
    search_request.page_size = 10
    search_request.visitor_id = '123456'
    search_request.page_size = 10
    print('---search request:---')
    print(search_request)
    return search_request

def search():
    if False:
        for i in range(10):
            print('nop')
    filter_ = '(colorFamilies: ANY("Black"))'
    search_request = get_search_request('Tee', filter_)
    search_response = SearchServiceClient().search(search_request)
    print('---search response---')
    if not search_response.results:
        print('The search operation returned no matching results.')
    else:
        print(search_response)
    return search_response
search()