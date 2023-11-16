import requests

def test_server(host_ip, http_port):
    if False:
        return 10
    '\n    Test that wordpress was setup on the minion correctly and returns a 200 after being installed\n    '
    resp = requests.get('http://{}:{}'.format(host_ip, http_port), headers={'Host': 'blog.manfred.io'})
    assert resp.status_code == 200