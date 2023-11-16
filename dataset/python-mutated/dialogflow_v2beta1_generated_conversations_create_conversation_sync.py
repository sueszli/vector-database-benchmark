from google.cloud import dialogflow_v2beta1

def sample_create_conversation():
    if False:
        return 10
    client = dialogflow_v2beta1.ConversationsClient()
    conversation = dialogflow_v2beta1.Conversation()
    conversation.conversation_profile = 'conversation_profile_value'
    request = dialogflow_v2beta1.CreateConversationRequest(parent='parent_value', conversation=conversation)
    response = client.create_conversation(request=request)
    print(response)