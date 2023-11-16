from google.cloud import policysimulator_v1

def sample_list_replay_results():
    if False:
        return 10
    client = policysimulator_v1.SimulatorClient()
    request = policysimulator_v1.ListReplayResultsRequest(parent='parent_value')
    page_result = client.list_replay_results(request=request)
    for response in page_result:
        print(response)