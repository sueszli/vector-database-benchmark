import google.auth
from google.cloud.retail import SearchRequest, SearchServiceClient
project_id = google.auth.default()[1]

def get_search_request(query: str, condition: str, boost_strength: float):
    if False:
        return 10
    default_search_placement = 'projects/' + project_id + '/locations/global/catalogs/default_catalog/placements/default_search'
    condition_boost_spec = SearchRequest.BoostSpec.ConditionBoostSpec()
    condition_boost_spec.condition = condition
    condition_boost_spec.boost = boost_strength
    boost_spec = SearchRequest.BoostSpec()
    boost_spec.condition_boost_specs = [condition_boost_spec]
    search_request = SearchRequest()
    search_request.placement = default_search_placement
    search_request.query = query
    search_request.visitor_id = '123456'
    search_request.boost_spec = boost_spec
    search_request.page_size = 10
    print('---search request---')
    print(search_request)
    return search_request

def search():
    if False:
        i = 10
        return i + 15
    condition = '(colorFamilies: ANY("Blue"))'
    boost = 0.0
    search_request = get_search_request('Tee', condition, boost)
    search_response = SearchServiceClient().search(search_request)
    print('---search response---')
    if not search_response.results:
        print('The search operation returned no matching results.')
    else:
        print(search_response)
    return search_response
search()