"""
Lookup the reverse DNS pointer records for one or more IP addresses.

 python  multi_reverse_lookup.py 127.0.0.1  192.0.2.100

IPADDRESS: An IPv4 or IPv6 address.
"""
import socket
import sys
from twisted.internet import defer, task
from twisted.names import client
from twisted.python import usage

class Options(usage.Options):
    synopsis = 'Usage: multi_reverse_lookup.py IPADDRESS [IPADDRESS]'

    def parseArgs(self, *addresses):
        if False:
            return 10
        self['addresses'] = addresses

def reverseNameFromIPv4Address(address):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a reverse domain name for the given IPv4 address.\n    '
    tokens = list(reversed(address.split('.'))) + ['in-addr', 'arpa', '']
    return '.'.join(tokens)

def reverseNameFromIPv6Address(address):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a reverse domain name for the given IPv6 address.\n    '
    fullHex = ''.join((f'{ord(c):02x}' for c in socket.inet_pton(socket.AF_INET6, address)))
    tokens = list(reversed(fullHex)) + ['ip6', 'arpa', '']
    return '.'.join(tokens)

def reverseNameFromIPAddress(address):
    if False:
        i = 10
        return i + 15
    '\n    Return a reverse domain name for the given IP address.\n    '
    try:
        socket.inet_pton(socket.AF_INET, address)
    except OSError:
        return reverseNameFromIPv6Address(address)
    else:
        return reverseNameFromIPv4Address(address)

def printResult(result):
    if False:
        return 10
    '\n    Print a comma separated list of reverse domain names and associated pointer\n    records.\n    '
    (answers, authority, additional) = result
    if answers:
        sys.stdout.write(', '.join((f'{a.name.name} IN {a.payload}' for a in answers)) + '\n')

def printSummary(results):
    if False:
        return 10
    '\n    Print a summary showing the total number of responses and queries.\n    '
    statuses = zip(*results)[0]
    sys.stdout.write(f'{statuses.count(True)} responses to {len(statuses)} queries' + '\n')

def main(reactor, *argv):
    if False:
        for i in range(10):
            print('nop')
    options = Options()
    try:
        options.parseOptions(argv)
    except usage.UsageError as errortext:
        sys.stderr.write(str(options) + '\n')
        sys.stderr.write(f'ERROR: {errortext}\n')
        raise SystemExit(1)
    pending = []
    for address in options['addresses']:
        pointerName = reverseNameFromIPAddress(address)
        result = client.lookupPointer(pointerName, timeout=(1,))
        result.addCallback(printResult)
        pending.append(result)
    allResults = defer.DeferredList(pending, consumeErrors=False)
    allResults.addCallback(printSummary)
    return allResults
if __name__ == '__main__':
    task.react(main, sys.argv[1:])