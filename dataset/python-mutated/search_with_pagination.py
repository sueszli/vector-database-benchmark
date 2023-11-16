import google.auth
from google.cloud.retail import SearchRequest, SearchServiceClient
project_id = google.auth.default()[1]

def get_search_request(query: str, page_size: int, offset: int, next_page_token: str):
    if False:
        for i in range(10):
            print('nop')
    default_search_placement = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/placements/default_search'
    search_request = SearchRequest()
    search_request.placement = default_search_placement
    search_request.visitor_id = '123456'
    search_request.query = query
    search_request.page_size = page_size
    search_request.offset = offset
    search_request.page_token = next_page_token
    print('---search request:---')
    print(search_request)
    return search_request

def search():
    if False:
        while True:
            i = 10
    page_size = 6
    offset = 0
    page_token = ''
    search_request_first_page = get_search_request('Hoodie', page_size, offset, page_token)
    search_response_first_page = SearchServiceClient().search(search_request_first_page)
    print('---search response---')
    if not search_response_first_page.results:
        print('The search operation returned no matching results.')
    else:
        print(search_response_first_page)
    return search_response_first_page
search()