from google.cloud import dialogflow_v2beta1

def sample_train_agent():
    if False:
        return 10
    client = dialogflow_v2beta1.AgentsClient()
    request = dialogflow_v2beta1.TrainAgentRequest(parent='parent_value')
    operation = client.train_agent(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)