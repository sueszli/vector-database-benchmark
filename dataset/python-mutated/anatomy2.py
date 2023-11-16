"""An addon using the abbreviated scripting syntax."""

def request(flow):
    if False:
        for i in range(10):
            print('nop')
    flow.request.headers['myheader'] = 'value'