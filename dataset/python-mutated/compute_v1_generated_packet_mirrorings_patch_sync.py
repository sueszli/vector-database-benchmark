from google.cloud import compute_v1

def sample_patch():
    if False:
        return 10
    client = compute_v1.PacketMirroringsClient()
    request = compute_v1.PatchPacketMirroringRequest(packet_mirroring='packet_mirroring_value', project='project_value', region='region_value')
    response = client.patch(request=request)
    print(response)