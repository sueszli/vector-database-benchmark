from google.cloud import contentwarehouse_v1

def sample_create_rule_set():
    if False:
        while True:
            i = 10
    client = contentwarehouse_v1.RuleSetServiceClient()
    request = contentwarehouse_v1.CreateRuleSetRequest(parent='parent_value')
    response = client.create_rule_set(request=request)
    print(response)