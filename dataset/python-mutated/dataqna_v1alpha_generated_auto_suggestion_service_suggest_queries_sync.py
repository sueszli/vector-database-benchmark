from google.cloud import dataqna_v1alpha

def sample_suggest_queries():
    if False:
        return 10
    client = dataqna_v1alpha.AutoSuggestionServiceClient()
    request = dataqna_v1alpha.SuggestQueriesRequest(parent='parent_value')
    response = client.suggest_queries(request=request)
    print(response)