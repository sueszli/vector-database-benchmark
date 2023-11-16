from google.cloud import policysimulator_v1

def sample_create_replay():
    if False:
        print('Hello World!')
    client = policysimulator_v1.SimulatorClient()
    request = policysimulator_v1.CreateReplayRequest(parent='parent_value')
    operation = client.create_replay(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)