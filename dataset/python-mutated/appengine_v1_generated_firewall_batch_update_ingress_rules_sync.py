from google.cloud import appengine_admin_v1

def sample_batch_update_ingress_rules():
    if False:
        for i in range(10):
            print('nop')
    client = appengine_admin_v1.FirewallClient()
    request = appengine_admin_v1.BatchUpdateIngressRulesRequest()
    response = client.batch_update_ingress_rules(request=request)
    print(response)