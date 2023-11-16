"""
Print the IP address for a given hostname. eg

 python gethostbyname.py www.google.com

This script does a host lookup using the default Twisted Names
resolver, a chained resolver, which attempts to lookup a name from:
 * local hosts file
 * memory cache of previous lookup results
 * system recursive DNS servers
"""
import sys
from twisted.internet.task import react
from twisted.names import client, error
from twisted.python import usage

class Options(usage.Options):
    synopsis = 'Usage: gethostbyname.py HOSTNAME'

    def parseArgs(self, hostname):
        if False:
            while True:
                i = 10
        self['hostname'] = hostname

def printResult(address, hostname):
    if False:
        while True:
            i = 10
    '\n    Print the IP address or an error message if an IP address was not\n    found.\n    '
    if address:
        sys.stdout.write(address + '\n')
    else:
        sys.stderr.write(f'ERROR: No IP addresses found for name {hostname!r}\n')

def printError(failure, hostname):
    if False:
        for i in range(10):
            print('nop')
    '\n    Print a friendly error message if the hostname could not be\n    resolved.\n    '
    failure.trap(error.DNSNameError)
    sys.stderr.write(f'ERROR: hostname not found {hostname!r}\n')

def main(reactor, *argv):
    if False:
        i = 10
        return i + 15
    options = Options()
    try:
        options.parseOptions(argv)
    except usage.UsageError as errortext:
        sys.stderr.write(str(options) + '\n')
        sys.stderr.write(f'ERROR: {errortext}\n')
        raise SystemExit(1)
    hostname = options['hostname']
    d = client.getHostByName(hostname)
    d.addCallback(printResult, hostname)
    d.addErrback(printError, hostname)
    return d
if __name__ == '__main__':
    react(main, sys.argv[1:])