from google.cloud.dialogflow_v2 import Agent, AgentsClient, SetAgentRequest
import google.protobuf.field_mask_pb2

def set_agent(project_id, display_name):
    if False:
        print('Hello World!')
    agents_client = AgentsClient()
    parent = agents_client.common_project_path(project_id)
    agent = Agent(parent=parent, display_name=display_name, default_language_code='en', time_zone='America/Los_Angeles')
    update_mask = google.protobuf.field_mask_pb2.FieldMask()
    update_mask.FromJsonString('displayName,defaultLanguageCode,timeZone')
    request = SetAgentRequest(agent=agent, update_mask=update_mask)
    return agents_client.set_agent(request=request)