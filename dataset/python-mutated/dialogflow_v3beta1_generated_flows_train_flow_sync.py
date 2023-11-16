from google.cloud import dialogflowcx_v3beta1

def sample_train_flow():
    if False:
        i = 10
        return i + 15
    client = dialogflowcx_v3beta1.FlowsClient()
    request = dialogflowcx_v3beta1.TrainFlowRequest(name='name_value')
    operation = client.train_flow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)