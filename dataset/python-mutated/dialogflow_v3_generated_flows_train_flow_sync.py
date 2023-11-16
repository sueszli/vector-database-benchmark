from google.cloud import dialogflowcx_v3

def sample_train_flow():
    if False:
        while True:
            i = 10
    client = dialogflowcx_v3.FlowsClient()
    request = dialogflowcx_v3.TrainFlowRequest(name='name_value')
    operation = client.train_flow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)