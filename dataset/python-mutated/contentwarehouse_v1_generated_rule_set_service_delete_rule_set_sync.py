from google.cloud import contentwarehouse_v1

def sample_delete_rule_set():
    if False:
        print('Hello World!')
    client = contentwarehouse_v1.RuleSetServiceClient()
    request = contentwarehouse_v1.DeleteRuleSetRequest(name='name_value')
    client.delete_rule_set(request=request)