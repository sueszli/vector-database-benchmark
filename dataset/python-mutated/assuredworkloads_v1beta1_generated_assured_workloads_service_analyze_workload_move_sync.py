from google.cloud import assuredworkloads_v1beta1

def sample_analyze_workload_move():
    if False:
        for i in range(10):
            print('nop')
    client = assuredworkloads_v1beta1.AssuredWorkloadsServiceClient()
    request = assuredworkloads_v1beta1.AnalyzeWorkloadMoveRequest(source='source_value', target='target_value')
    response = client.analyze_workload_move(request=request)
    print(response)