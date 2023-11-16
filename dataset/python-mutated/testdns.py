"""
Print the Address records, Mail-Exchanger records and the Nameserver
records for the given domain name. eg

 python testdns.py google.com
"""
import sys
from twisted.internet import defer
from twisted.internet.task import react
from twisted.names import client, dns, error
from twisted.python import usage

class Options(usage.Options):
    synopsis = 'Usage: testdns.py DOMAINNAME'

    def parseArgs(self, domainname):
        if False:
            return 10
        self['domainname'] = domainname

def formatRecords(records, heading):
    if False:
        i = 10
        return i + 15
    '\n    Extract only the answer records and return them as a neatly\n    formatted string beneath the given heading.\n    '
    (answers, authority, additional) = records
    lines = ['# ' + heading]
    for a in answers:
        line = [a.name, dns.QUERY_CLASSES.get(a.cls, 'UNKNOWN (%d)' % (a.cls,)), a.payload]
        lines.append(' '.join((str(word) for word in line)))
    return '\n'.join((line for line in lines))

def printResults(results, domainname):
    if False:
        i = 10
        return i + 15
    '\n    Print the formatted results for each DNS record type.\n    '
    sys.stdout.write(f'# Domain Summary for {domainname!r}\n')
    sys.stdout.write('\n\n'.join(results) + '\n')

def printError(failure, domainname):
    if False:
        i = 10
        return i + 15
    '\n    Print a friendly error message if the hostname could not be\n    resolved.\n    '
    failure.trap(defer.FirstError)
    failure = failure.value.subFailure
    failure.trap(error.DNSNameError)
    sys.stderr.write(f'ERROR: domain name not found {domainname!r}\n')

def main(reactor, *argv):
    if False:
        print('Hello World!')
    options = Options()
    try:
        options.parseOptions(argv)
    except usage.UsageError as errortext:
        sys.stderr.write(str(options) + '\n')
        sys.stderr.write(f'ERROR: {errortext}\n')
        raise SystemExit(1)
    domainname = options['domainname']
    r = client.Resolver('/etc/resolv.conf')
    d = defer.gatherResults([r.lookupAddress(domainname).addCallback(formatRecords, 'Addresses'), r.lookupMailExchange(domainname).addCallback(formatRecords, 'Mail Exchangers'), r.lookupNameservers(domainname).addCallback(formatRecords, 'Nameservers')], consumeErrors=True)
    d.addCallback(printResults, domainname)
    d.addErrback(printError, domainname)
    return d
if __name__ == '__main__':
    react(main, sys.argv[1:])