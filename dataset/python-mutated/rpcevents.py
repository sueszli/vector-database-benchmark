import click
import logging
from twisted.internet.defer import inlineCallbacks
from golem.core.common import config_logging
from golem.node import OptNode
from golem.rpc.mapping.rpcmethodnames import NAMESPACES
from golem.rpc.session import Session, WebSocketAddress

class EventLoggingSession(Session):

    def __init__(self, logger, address, methods=None, events=None):
        if False:
            while True:
                i = 10
        super(EventLoggingSession, self).__init__(address, methods, events)
        self.logger = logger

    @inlineCallbacks
    def onJoin(self, details):
        if False:
            i = 10
            return i + 15
        yield super(EventLoggingSession, self).onJoin(details)
        self.logger.info('| onJoin(%s)', details)

    def onUserError(self, fail, msg):
        if False:
            while True:
                i = 10
        super(EventLoggingSession, self).onUserError(fail, msg)
        self.logger.error('| onUserError %s %s', fail, msg)

    def onConnect(self):
        if False:
            while True:
                i = 10
        super(EventLoggingSession, self).onConnect()
        self.logger.info('| onConnect')

    def onClose(self, wasClean):
        if False:
            i = 10
            return i + 15
        super(EventLoggingSession, self).onClose(wasClean)
        self.logger.info('| onClose(wasClean=%s)', wasClean)

    def onLeave(self, details):
        if False:
            i = 10
            return i + 15
        super(EventLoggingSession, self).onLeave(details)
        self.logger.info('| onLeave(details=%s)', details)

    def onDisconnect(self):
        if False:
            for i in range(10):
                print('nop')
        super(EventLoggingSession, self).onDisconnect()
        self.logger.info('| onDisconnect')

def build_handler(logger, evt_name):
    if False:
        return 10

    def handler(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        logger.info('%s %s %s', evt_name, args, kwargs)
    return handler

def build_handlers(logger):
    if False:
        return 10
    handlers = set()
    for ns in NAMESPACES:
        for (prop, value) in ns.__dict__.items():
            if is_event(prop, value):
                entry = (build_handler(logger, value), value)
                handlers.add(entry)
    return handlers

def is_event(prop, value):
    if False:
        print('Hello World!')
    return not prop.startswith('_') and isinstance(value, basestring) and value.startswith('evt.')

@click.command()
@click.option('--datadir', '-d', type=click.Path())
@click.option('--rpc-address', '-r', multiple=False, callback=OptNode.parse_rpc_address, help='RPC server address: <ip_addr>:<port>')
def main(datadir, rpc_address):
    if False:
        while True:
            i = 10
    from twisted.internet import reactor
    if rpc_address:
        host = rpc_address.address
        port = rpc_address.port
    else:
        host = 'localhost'
        port = 61000
    config_logging(datadir=datadir)
    logger = logging.getLogger('events')
    address = WebSocketAddress(host, port, realm=u'golem')
    events = build_handlers(logger)
    rpc_session = EventLoggingSession(logger, address, events=events)
    rpc_session.connect(auto_reconnect=True)
    reactor.run()
if __name__ == '__main__':
    main()