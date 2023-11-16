from google.cloud import clouddms_v1

def sample_get_mapping_rule():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.GetMappingRuleRequest(name='name_value')
    response = client.get_mapping_rule(request=request)
    print(response)