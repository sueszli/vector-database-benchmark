from google.cloud import policysimulator_v1

def sample_get_replay():
    if False:
        return 10
    client = policysimulator_v1.SimulatorClient()
    request = policysimulator_v1.GetReplayRequest(name='name_value')
    response = client.get_replay(request=request)
    print(response)