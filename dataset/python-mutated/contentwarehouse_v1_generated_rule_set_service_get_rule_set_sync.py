from google.cloud import contentwarehouse_v1

def sample_get_rule_set():
    if False:
        while True:
            i = 10
    client = contentwarehouse_v1.RuleSetServiceClient()
    request = contentwarehouse_v1.GetRuleSetRequest(name='name_value')
    response = client.get_rule_set(request=request)
    print(response)