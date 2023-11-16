"""
RPCMethods is a dictionary listing RPC transports currently supported
by this client.

Args:
    function: the function whose parameter list will be examined
    excluded_args: function arguments that are NOT to be added to the dictionary (sequence of strings)
    options: result of command argument parsing (optparse.Values)
"""
RPCMethods = {'thrift': 'Apache Thrift'}
"\nBase class for RPC transport clients\n\nMethods that all RPC clients should implement include:\n\n    def newConnection(host,port): Method for re-establishing a new client\n        connection to a different host / port\n\n    def properties([]): Given a list of ControlPort property names,\n        or an empty list to specify all currently registered properties,\n        this method returns a dictionary of metadata describing the\n        the specified properties. The dictionary key contains the name\n        of each returned properties.\n\n    def getKnobs([]): Given a list of ControlPort property names,\n        or an empty list to specify all currently registered properties,\n        this method returns a dictionary of the current value of\n        the specified properties.\n\n    def getRe([]): Given a list of regular expression strings,\n        this method returns a dictionary of the current value of\n        the all properties with names that match the specified\n        expressions.\n\n    def setKnobs({}): Given a dictionary of ControlPort property\n        key / value pairs, this method requests that ControlPort\n        attempt to set the specified named properties to the\n        value given. Success in setting each property to the\n        value specified requires that the property be registered\n        as a 'setable' ControlPort property, that the client have the\n        requisite privilege level to set the property, and\n        the underlying Block's implementation in handling\n        the set request.\n\nArgs:\n    method: name of the RPC transport\n    port: port number of the connection\n    host: hostname of the connection\n"

class RPCConnection(object):

    def __init__(self, method, port, host=None):
        if False:
            while True:
                i = 10
        (self.method, self.port) = (method, port)
        if host is None:
            self.host = '127.0.0.1'
        else:
            self.host = host

    def __str__(self):
        if False:
            print('Hello World!')
        return '%s connection on %s:%s' % (self.getName(), self.getHost(), self.getPort())

    def getName(self):
        if False:
            i = 10
            return i + 15
        return RPCMethods[self.method]

    def getHost(self):
        if False:
            while True:
                i = 10
        return self.host

    def getPort(self):
        if False:
            i = 10
            return i + 15
        return self.port

    def newConnection(self, host=None, port=None):
        if False:
            return 10
        raise NotImplementedError()

    def properties(self, *args):
        if False:
            while True:
                i = 10
        raise NotImplementedError()

    def getKnobs(self, *args):
        if False:
            return 10
        raise NotImplementedError()

    def getRe(self, *args):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def postMessage(self, *args):
        if False:
            return 10
        raise NotImplementedError()

    def setKnobs(self, *args):
        if False:
            return 10
        raise NotImplementedError()

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def printProperties(self, props):
        if False:
            return 10
        raise NotImplementedError()