from google.cloud import recommender_v1

def sample_mark_insight_accepted():
    if False:
        print('Hello World!')
    client = recommender_v1.RecommenderClient()
    request = recommender_v1.MarkInsightAcceptedRequest(name='name_value', etag='etag_value')
    response = client.mark_insight_accepted(request=request)
    print(response)