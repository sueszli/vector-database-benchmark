import logging

def modify(chunks):
    if False:
        return 10
    for chunk in chunks:
        yield chunk.replace(b'foo', b'bar')

def running():
    if False:
        i = 10
        return i + 15
    logging.info('stream_modify running')

def responseheaders(flow):
    if False:
        while True:
            i = 10
    flow.response.stream = modify