from google.cloud.dialogflowcx_v3.services.intents import IntentsClient
from google.protobuf import field_mask_pb2

def update_intent(project_id, agent_id, intent_id, location, displayName):
    if False:
        return 10
    intents_client = IntentsClient()
    intent_name = intents_client.intent_path(project_id, location, agent_id, intent_id)
    intent = intents_client.get_intent(request={'name': intent_name})
    intent.display_name = displayName
    update_mask = field_mask_pb2.FieldMask(paths=['display_name'])
    response = intents_client.update_intent(intent=intent, update_mask=update_mask)
    return response