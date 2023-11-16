from typing import List
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine

def multi_turn_search_sample(project_id: str, location: str, data_store_id: str, search_queries: List[str]) -> List[discoveryengine.ConverseConversationResponse]:
    if False:
        while True:
            i = 10
    client_options = ClientOptions(api_endpoint=f'{location}-discoveryengine.googleapis.com') if location != 'global' else None
    client = discoveryengine.ConversationalSearchServiceClient(client_options=client_options)
    conversation = client.create_conversation(parent=client.data_store_path(project=project_id, location=location, data_store=data_store_id), conversation=discoveryengine.Conversation())
    responses: List[discoveryengine.ConverseConversationResponse] = []
    for search_query in search_queries:
        request = discoveryengine.ConverseConversationRequest(name=conversation.name, query=discoveryengine.TextInput(input=search_query), serving_config=client.serving_config_path(project=project_id, location=location, data_store=data_store_id, serving_config='default_config'), summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(summary_result_count=3, include_citations=True))
        response = client.converse_conversation(request)
        print(f'Reply: {response.reply.summary.summary_text}\n')
        for (i, result) in enumerate(response.search_results, 1):
            result_data = result.document.derived_struct_data
            print(f'[{i}]')
            print(f"Link: {result_data['link']}")
            print(f"First Snippet: {result_data['snippets'][0]['snippet']}")
            print(f"First Extractive Answer: \n\tPage: {result_data['extractive_answers'][0]['pageNumber']}\n\tContent: {result_data['extractive_answers'][0]['content']}\n\n")
        print('\n\n')
        responses.append(response)
    return responses