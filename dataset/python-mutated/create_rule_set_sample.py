from google.cloud import contentwarehouse

def create_rule_set(project_number: str, location: str) -> None:
    if False:
        print('Hello World!')
    client = contentwarehouse.RuleSetServiceClient()
    parent = client.common_location_path(project=project_number, location=location)
    actions = contentwarehouse.Action(delete_document_action=contentwarehouse.DeleteDocumentAction(enable_hard_delete=True))
    rules = contentwarehouse.Rule(trigger_type='ON_CREATE', condition="documentType == 'W9' && STATE =='CA'", actions=[actions])
    rule_set = contentwarehouse.RuleSet(description='W9: Basic validation check rules.', source='My Organization', rules=[rules])
    request = contentwarehouse.CreateRuleSetRequest(parent=parent, rule_set=rule_set)
    response = client.create_rule_set(request=request)
    print(f'Rule Set Created: {response}')
    request = contentwarehouse.ListRuleSetsRequest(parent=parent)
    page_result = client.list_rule_sets(request=request)
    for response in page_result:
        print(f'Rule Sets: {response}')