from twisted.internet import reactor

def gotIP(ip):
    if False:
        print('Hello World!')
    print("IP of 'localhost' is", ip)
    reactor.stop()
reactor.resolve('localhost').addCallback(gotIP)
reactor.run()