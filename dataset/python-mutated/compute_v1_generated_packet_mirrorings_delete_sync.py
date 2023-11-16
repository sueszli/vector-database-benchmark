from google.cloud import compute_v1

def sample_delete():
    if False:
        print('Hello World!')
    client = compute_v1.PacketMirroringsClient()
    request = compute_v1.DeletePacketMirroringRequest(packet_mirroring='packet_mirroring_value', project='project_value', region='region_value')
    response = client.delete(request=request)
    print(response)