import os
from discoveryengine import multi_turn_search_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
search_queries = ['What is Google?', 'What was their revenue in 2021?']

def test_multi_turn_search():
    if False:
        while True:
            i = 10
    location = 'global'
    data_store_id = 'alphabet-earnings-reports_1697472013405'
    responses = multi_turn_search_sample.multi_turn_search_sample(project_id=project_id, location=location, data_store_id=data_store_id, search_queries=search_queries)
    assert responses
    for response in responses:
        assert response.reply
        assert response.conversation
        assert response.search_results
        for result in response.search_results:
            assert result.document.name