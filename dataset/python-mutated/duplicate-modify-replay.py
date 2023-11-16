"""Take incoming HTTP requests and replay them with modified parameters."""
from mitmproxy import ctx

def request(flow):
    if False:
        print('Hello World!')
    if flow.is_replay == 'request':
        return
    flow = flow.copy()
    if 'view' in ctx.master.addons:
        ctx.master.commands.call('view.flows.duplicate', [flow])
    flow.request.path = '/changed'
    ctx.master.commands.call('replay.client', [flow])