from google.cloud import dialogflow_v2

def sample_generate_stateless_summary():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ConversationsClient()
    stateless_conversation = dialogflow_v2.MinimalConversation()
    stateless_conversation.messages.content = 'content_value'
    stateless_conversation.parent = 'parent_value'
    conversation_profile = dialogflow_v2.ConversationProfile()
    conversation_profile.display_name = 'display_name_value'
    request = dialogflow_v2.GenerateStatelessSummaryRequest(stateless_conversation=stateless_conversation, conversation_profile=conversation_profile)
    response = client.generate_stateless_summary(request=request)
    print(response)