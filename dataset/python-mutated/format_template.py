def handler(event, context):
    if False:
        print('Hello World!')
    parameters = event['templateParameterValues']
    fragment = walk(event['fragment'], parameters)
    resp = {'requestId': event['requestId'], 'status': 'success', 'fragment': fragment}
    return resp

def walk(node, context):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(node, dict):
        return {k: walk(v, context) for (k, v) in node.items()}
    elif isinstance(node, list):
        return [walk(elem, context) for elem in node]
    elif isinstance(node, str):
        return node.format(**context)
    else:
        return node