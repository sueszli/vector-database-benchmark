from google.cloud import compute_v1

def sample_insert():
    if False:
        print('Hello World!')
    client = compute_v1.PacketMirroringsClient()
    request = compute_v1.InsertPacketMirroringRequest(project='project_value', region='region_value')
    response = client.insert(request=request)
    print(response)