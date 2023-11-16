from google.cloud import appengine_admin_v1

def sample_update_ingress_rule():
    if False:
        while True:
            i = 10
    client = appengine_admin_v1.FirewallClient()
    request = appengine_admin_v1.UpdateIngressRuleRequest()
    response = client.update_ingress_rule(request=request)
    print(response)