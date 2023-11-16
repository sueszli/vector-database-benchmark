from calculus.base_3 import Calculation
from twisted.internet import protocol
from twisted.protocols import basic

class CalculationProxy:

    def __init__(self):
        if False:
            print('Hello World!')
        self.calc = Calculation()
        for m in ['add', 'subtract', 'multiply', 'divide']:
            setattr(self, f'remote_{m}', getattr(self.calc, m))

class RemoteCalculationProtocol(basic.LineReceiver):

    def __init__(self):
        if False:
            return 10
        self.proxy = CalculationProxy()

    def lineReceived(self, line):
        if False:
            for i in range(10):
                print('nop')
        (op, a, b) = line.decode('utf-8').split()
        a = int(a)
        b = int(b)
        op = getattr(self.proxy, f'remote_{op}')
        result = op(a, b)
        self.sendLine(str(result).encode('utf-8'))

class RemoteCalculationFactory(protocol.Factory):
    protocol = RemoteCalculationProtocol

def main():
    if False:
        i = 10
        return i + 15
    import sys
    from twisted.internet import reactor
    from twisted.python import log
    log.startLogging(sys.stdout)
    reactor.listenTCP(0, RemoteCalculationFactory())
    reactor.run()
if __name__ == '__main__':
    main()