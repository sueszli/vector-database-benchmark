from calculus.base_3 import Calculation
from twisted.internet import protocol
from twisted.protocols import basic
from twisted.python import log

class CalculationProxy:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.calc = Calculation()
        for m in ['add', 'subtract', 'multiply', 'divide']:
            setattr(self, f'remote_{m}', getattr(self.calc, m))

class RemoteCalculationProtocol(basic.LineReceiver):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.proxy = CalculationProxy()

    def lineReceived(self, line):
        if False:
            i = 10
            return i + 15
        (op, a, b) = line.decode('utf-8').split()
        op = getattr(self.proxy, 'remote_{}'.format(op))
        try:
            result = op(a, b)
        except TypeError:
            log.err()
            self.sendLine(b'error')
        else:
            self.sendLine(str(result).encode('utf-8'))

class RemoteCalculationFactory(protocol.Factory):
    protocol = RemoteCalculationProtocol

def main():
    if False:
        return 10
    import sys
    from twisted.internet import reactor
    from twisted.python import log
    log.startLogging(sys.stdout)
    reactor.listenTCP(0, RemoteCalculationFactory())
    reactor.run()
if __name__ == '__main__':
    main()