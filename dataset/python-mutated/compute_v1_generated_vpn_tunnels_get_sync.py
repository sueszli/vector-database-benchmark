from google.cloud import compute_v1

def sample_get():
    if False:
        return 10
    client = compute_v1.VpnTunnelsClient()
    request = compute_v1.GetVpnTunnelRequest(project='project_value', region='region_value', vpn_tunnel='vpn_tunnel_value')
    response = client.get(request=request)
    print(response)