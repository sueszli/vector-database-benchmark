from google.cloud import compute_v1

def sample_delete():
    if False:
        i = 10
        return i + 15
    client = compute_v1.VpnTunnelsClient()
    request = compute_v1.DeleteVpnTunnelRequest(project='project_value', region='region_value', vpn_tunnel='vpn_tunnel_value')
    response = client.delete(request=request)
    print(response)