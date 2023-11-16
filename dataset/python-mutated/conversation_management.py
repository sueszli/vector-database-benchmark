"""Dialogflow API Python sample showing how to manage Conversations.
"""
from google.cloud import dialogflow_v2beta1 as dialogflow

def create_conversation(project_id, conversation_profile_id):
    if False:
        i = 10
        return i + 15
    'Creates a conversation with given values\n\n    Args:\n        project_id:  The GCP project linked with the conversation.\n        conversation_profile_id: The conversation profile id used to create\n        conversation.'
    client = dialogflow.ConversationsClient()
    conversation_profile_client = dialogflow.ConversationProfilesClient()
    project_path = client.common_project_path(project_id)
    conversation_profile_path = conversation_profile_client.conversation_profile_path(project_id, conversation_profile_id)
    conversation = {'conversation_profile': conversation_profile_path}
    response = client.create_conversation(parent=project_path, conversation=conversation)
    print('Life Cycle State: {}'.format(response.lifecycle_state))
    print('Conversation Profile Name: {}'.format(response.conversation_profile))
    print('Name: {}'.format(response.name))
    return response

def get_conversation(project_id, conversation_id):
    if False:
        i = 10
        return i + 15
    'Gets a specific conversation profile.\n\n    Args:\n        project_id: The GCP project linked with the conversation.\n        conversation_id: Id of the conversation.'
    client = dialogflow.ConversationsClient()
    conversation_path = client.conversation_path(project_id, conversation_id)
    response = client.get_conversation(name=conversation_path)
    print('Life Cycle State: {}'.format(response.lifecycle_state))
    print('Conversation Profile Name: {}'.format(response.conversation_profile))
    print('Name: {}'.format(response.name))
    return response

def complete_conversation(project_id, conversation_id):
    if False:
        print('Hello World!')
    'Completes the specified conversation. Finished conversations are purged from the database after 30 days.\n\n    Args:\n        project_id: The GCP project linked with the conversation.\n        conversation_id: Id of the conversation.'
    client = dialogflow.ConversationsClient()
    conversation_path = client.conversation_path(project_id, conversation_id)
    conversation = client.complete_conversation(name=conversation_path)
    print('Completed Conversation.')
    print('Life Cycle State: {}'.format(conversation.lifecycle_state))
    print('Conversation Profile Name: {}'.format(conversation.conversation_profile))
    print('Name: {}'.format(conversation.name))
    return conversation