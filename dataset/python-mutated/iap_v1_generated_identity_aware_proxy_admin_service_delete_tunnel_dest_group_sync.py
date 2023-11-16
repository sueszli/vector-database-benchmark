from google.cloud import iap_v1

def sample_delete_tunnel_dest_group():
    if False:
        while True:
            i = 10
    client = iap_v1.IdentityAwareProxyAdminServiceClient()
    request = iap_v1.DeleteTunnelDestGroupRequest(name='name_value')
    client.delete_tunnel_dest_group(request=request)