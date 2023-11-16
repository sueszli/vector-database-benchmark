from google.cloud import compute_v1

def sample_get():
    if False:
        while True:
            i = 10
    client = compute_v1.PacketMirroringsClient()
    request = compute_v1.GetPacketMirroringRequest(packet_mirroring='packet_mirroring_value', project='project_value', region='region_value')
    response = client.get(request=request)
    print(response)