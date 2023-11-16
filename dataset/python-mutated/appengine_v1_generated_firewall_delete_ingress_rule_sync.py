from google.cloud import appengine_admin_v1

def sample_delete_ingress_rule():
    if False:
        return 10
    client = appengine_admin_v1.FirewallClient()
    request = appengine_admin_v1.DeleteIngressRuleRequest()
    client.delete_ingress_rule(request=request)