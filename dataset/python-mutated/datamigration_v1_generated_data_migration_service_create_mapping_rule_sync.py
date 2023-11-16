from google.cloud import clouddms_v1

def sample_create_mapping_rule():
    if False:
        i = 10
        return i + 15
    client = clouddms_v1.DataMigrationServiceClient()
    mapping_rule = clouddms_v1.MappingRule()
    mapping_rule.single_entity_rename.new_name = 'new_name_value'
    mapping_rule.rule_scope = 'DATABASE_ENTITY_TYPE_DATABASE'
    mapping_rule.rule_order = 1075
    request = clouddms_v1.CreateMappingRuleRequest(parent='parent_value', mapping_rule_id='mapping_rule_id_value', mapping_rule=mapping_rule)
    response = client.create_mapping_rule(request=request)
    print(response)