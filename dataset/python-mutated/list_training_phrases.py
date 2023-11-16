def list_training_phrases(project_id, agent_id, intent_id, location):
    if False:
        return 10
    'Returns all training phrases for a specified intent.'
    from google.cloud import dialogflowcx
    intent_client = dialogflowcx.IntentsClient()
    intent_name = intent_client.intent_path(project_id, location, agent_id, intent_id)
    get_intent_request = dialogflowcx.GetIntentRequest(name=intent_name)
    intent = intent_client.get_intent(get_intent_request)
    for phrase in intent.training_phrases:
        print(phrase)
    return intent.training_phrases