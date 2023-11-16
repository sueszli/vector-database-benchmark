from google.cloud import migrationcenter_v1

def sample_list_preference_sets():
    if False:
        return 10
    client = migrationcenter_v1.MigrationCenterClient()
    request = migrationcenter_v1.ListPreferenceSetsRequest(parent='parent_value')
    page_result = client.list_preference_sets(request=request)
    for response in page_result:
        print(response)