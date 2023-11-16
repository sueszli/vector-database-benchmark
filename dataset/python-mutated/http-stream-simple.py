"""
Select which responses should be streamed.

Enable response streaming for all HTTP flows.
This is equivalent to passing `--set stream_large_bodies=1` to mitmproxy.
"""

def responseheaders(flow):
    if False:
        return 10
    '\n    Enables streaming for all responses.\n    This is equivalent to passing `--set stream_large_bodies=1` to mitmproxy.\n    '
    flow.response.stream = True