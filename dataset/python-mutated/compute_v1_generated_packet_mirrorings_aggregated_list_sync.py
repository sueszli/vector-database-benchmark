from google.cloud import compute_v1

def sample_aggregated_list():
    if False:
        while True:
            i = 10
    client = compute_v1.PacketMirroringsClient()
    request = compute_v1.AggregatedListPacketMirroringsRequest(project='project_value')
    page_result = client.aggregated_list(request=request)
    for response in page_result:
        print(response)