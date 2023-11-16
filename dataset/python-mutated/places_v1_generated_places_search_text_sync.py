from google.maps import places_v1

def sample_search_text():
    if False:
        i = 10
        return i + 15
    client = places_v1.PlacesClient()
    request = places_v1.SearchTextRequest(text_query='text_query_value')
    response = client.search_text(request=request)
    print(response)