from google.cloud import dialogflow_v2

def sample_create_conversation():
    if False:
        for i in range(10):
            print('nop')
    client = dialogflow_v2.ConversationsClient()
    conversation = dialogflow_v2.Conversation()
    conversation.conversation_profile = 'conversation_profile_value'
    request = dialogflow_v2.CreateConversationRequest(parent='parent_value', conversation=conversation)
    response = client.create_conversation(request=request)
    print(response)