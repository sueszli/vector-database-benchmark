from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        return 10
    client = compute_v1.PacketMirroringsClient()
    request = compute_v1.TestIamPermissionsPacketMirroringRequest(project='project_value', region='region_value', resource='resource_value')
    response = client.test_iam_permissions(request=request)
    print(response)