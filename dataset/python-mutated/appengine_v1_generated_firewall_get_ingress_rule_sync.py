from google.cloud import appengine_admin_v1

def sample_get_ingress_rule():
    if False:
        print('Hello World!')
    client = appengine_admin_v1.FirewallClient()
    request = appengine_admin_v1.GetIngressRuleRequest()
    response = client.get_ingress_rule(request=request)
    print(response)