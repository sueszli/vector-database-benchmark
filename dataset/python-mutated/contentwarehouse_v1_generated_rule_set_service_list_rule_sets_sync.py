from google.cloud import contentwarehouse_v1

def sample_list_rule_sets():
    if False:
        while True:
            i = 10
    client = contentwarehouse_v1.RuleSetServiceClient()
    request = contentwarehouse_v1.ListRuleSetsRequest(parent='parent_value')
    page_result = client.list_rule_sets(request=request)
    for response in page_result:
        print(response)