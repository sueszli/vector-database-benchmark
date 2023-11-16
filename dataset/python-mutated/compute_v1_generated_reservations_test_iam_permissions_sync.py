from google.cloud import compute_v1

def sample_test_iam_permissions():
    if False:
        return 10
    client = compute_v1.ReservationsClient()
    request = compute_v1.TestIamPermissionsReservationRequest(project='project_value', resource='resource_value', zone='zone_value')
    response = client.test_iam_permissions(request=request)
    print(response)