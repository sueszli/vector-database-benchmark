from google.cloud import appengine_admin_v1

def sample_create_ingress_rule():
    if False:
        for i in range(10):
            print('nop')
    client = appengine_admin_v1.FirewallClient()
    request = appengine_admin_v1.CreateIngressRuleRequest()
    response = client.create_ingress_rule(request=request)
    print(response)