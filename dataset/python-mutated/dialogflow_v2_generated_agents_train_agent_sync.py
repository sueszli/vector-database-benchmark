from google.cloud import dialogflow_v2

def sample_train_agent():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.AgentsClient()
    request = dialogflow_v2.TrainAgentRequest(parent='parent_value')
    operation = client.train_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)