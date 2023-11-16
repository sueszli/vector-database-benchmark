def execute(event_name, client, server, handler, config, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    server.info('Event: {}, client={}, server={}, handler={}, config={}, kwargs={}'.format(event_name, client, server, handler, config, kwargs))