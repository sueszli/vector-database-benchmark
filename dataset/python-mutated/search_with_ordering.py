import google.auth
from google.cloud.retail import SearchRequest, SearchServiceClient
project_id = google.auth.default()[1]

def get_search_request(query: str, order: str):
    if False:
        i = 10
        return i + 15
    default_search_placement = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/placements/default_search'
    search_request = SearchRequest()
    search_request.placement = default_search_placement
    search_request.query = query
    search_request.order_by = order
    search_request.visitor_id = '123456'
    search_request.page_size = 10
    print('---search request---')
    print(search_request)
    return search_request

def search():
    if False:
        i = 10
        return i + 15
    order = 'price desc'
    search_request = get_search_request('Hoodie', order)
    search_response = SearchServiceClient().search(search_request)
    print('---search response---')
    if not search_response.results:
        print('The search operation returned no matching results.')
    else:
        print(search_response)
    return search_response
search()