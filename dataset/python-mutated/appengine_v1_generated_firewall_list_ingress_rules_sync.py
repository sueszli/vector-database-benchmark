from google.cloud import appengine_admin_v1

def sample_list_ingress_rules():
    if False:
        i = 10
        return i + 15
    client = appengine_admin_v1.FirewallClient()
    request = appengine_admin_v1.ListIngressRulesRequest()
    page_result = client.list_ingress_rules(request=request)
    for response in page_result:
        print(response)