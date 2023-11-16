def tcp_message(flow):
    if False:
        while True:
            i = 10
    message = flow.messages[-1]
    if not message.from_client:
        message.content = message.content.replace(b'foo', b'bar')