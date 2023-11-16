"""
Print the SRV records for a given DOMAINNAME eg

 python dns-service.py xmpp-client tcp gmail.com

SERVICE: the symbolic name of the desired service.

PROTO: the transport protocol of the desired service; this is usually
       either TCP or UDP.

DOMAINNAME: the domain name for which this record is valid.
"""
import sys
from twisted.internet.task import react
from twisted.names import client, error
from twisted.python import usage

class Options(usage.Options):
    synopsis = 'Usage: dns-service.py SERVICE PROTO DOMAINNAME'

    def parseArgs(self, service, proto, domainname):
        if False:
            print('Hello World!')
        self['service'] = service
        self['proto'] = proto
        self['domainname'] = domainname

def printResult(records, domainname):
    if False:
        print('Hello World!')
    '\n    Print the SRV records for the domainname or an error message if no\n    SRV records were found.\n    '
    (answers, authority, additional) = records
    if answers:
        sys.stdout.write(domainname + ' IN \n ' + '\n '.join((str(x.payload) for x in answers)) + '\n')
    else:
        sys.stderr.write(f'ERROR: No SRV records found for name {domainname!r}\n')

def printError(failure, domainname):
    if False:
        print('Hello World!')
    '\n    Print a friendly error message if the domainname could not be\n    resolved.\n    '
    failure.trap(error.DNSNameError)
    sys.stderr.write(f'ERROR: domain name not found {domainname!r}\n')

def main(reactor, *argv):
    if False:
        while True:
            i = 10
    options = Options()
    try:
        options.parseOptions(argv)
    except usage.UsageError as errortext:
        sys.stderr.write(str(options) + '\n')
        sys.stderr.write(f'ERROR: {errortext}\n')
        raise SystemExit(1)
    resolver = client.Resolver('/etc/resolv.conf')
    domainname = '_%(service)s._%(proto)s.%(domainname)s' % options
    d = resolver.lookupService(domainname)
    d.addCallback(printResult, domainname)
    d.addErrback(printError, domainname)
    return d
if __name__ == '__main__':
    react(main, sys.argv[1:])