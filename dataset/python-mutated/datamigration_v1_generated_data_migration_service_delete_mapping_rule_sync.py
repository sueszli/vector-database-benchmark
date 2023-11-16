from google.cloud import clouddms_v1

def sample_delete_mapping_rule():
    if False:
        i = 10
        return i + 15
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.DeleteMappingRuleRequest(name='name_value')
    client.delete_mapping_rule(request=request)