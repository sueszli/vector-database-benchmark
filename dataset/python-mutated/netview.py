from __future__ import division
from __future__ import print_function
import sys
import argparse
import logging
import socket
from threading import Thread, Event
from queue import Queue
from time import sleep
from impacket.examples import logger
from impacket.examples.utils import parse_credentials
from impacket import version
from impacket.smbconnection import SessionError
from impacket.dcerpc.v5 import transport, wkst, srvs, samr
from impacket.dcerpc.v5.ndr import NULL
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.nt_errors import STATUS_MORE_ENTRIES
machinesAliveQueue = Queue()
machinesDownQueue = Queue()
myIP = None

def checkMachines(machines, stopEvent, singlePass=False):
    if False:
        for i in range(10):
            print('nop')
    origLen = len(machines)
    deadMachines = machines
    done = False
    while not done:
        if stopEvent.is_set():
            done = True
            break
        for machine in deadMachines:
            s = socket.socket()
            try:
                s = socket.create_connection((machine, 445), 2)
                global myIP
                myIP = s.getsockname()[0]
                s.close()
                machinesAliveQueue.put(machine)
            except Exception as e:
                logging.debug('%s: not alive (%s)' % (machine, e))
                pass
            else:
                logging.debug('%s: alive!' % machine)
                deadMachines.remove(machine)
            if stopEvent.is_set():
                done = True
                break
        logging.debug('up: %d, down: %d, total: %d' % (origLen - len(deadMachines), len(deadMachines), origLen))
        if singlePass is True:
            done = True
        if not done:
            sleep(10)
            while machinesDownQueue.empty() is False:
                deadMachines.append(machinesDownQueue.get())

class USERENUM:

    def __init__(self, username='', password='', domain='', hashes=None, aesKey=None, doKerberos=False, options=None):
        if False:
            for i in range(10):
                print('nop')
        self.__username = username
        self.__password = password
        self.__domain = domain
        self.__lmhash = ''
        self.__nthash = ''
        self.__aesKey = aesKey
        self.__doKerberos = doKerberos
        self.__kdcHost = options.dc_ip
        self.__options = options
        self.__machinesList = list()
        self.__targets = dict()
        self.__filterUsers = None
        self.__targetsThreadEvent = None
        self.__targetsThread = None
        self.__maxConnections = int(options.max_connections)
        if hashes is not None:
            (self.__lmhash, self.__nthash) = hashes.split(':')

    def getDomainMachines(self):
        if False:
            return 10
        if self.__kdcHost is not None:
            domainController = self.__kdcHost
        elif self.__domain != '':
            domainController = self.__domain
        else:
            raise Exception('A domain is needed!')
        logging.info("Getting machine's list from %s" % domainController)
        rpctransport = transport.SMBTransport(domainController, 445, '\\samr', self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey, doKerberos=self.__doKerberos, kdcHost=self.__kdcHost)
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(samr.MSRPC_UUID_SAMR)
        try:
            resp = samr.hSamrConnect(dce)
            serverHandle = resp['ServerHandle']
            resp = samr.hSamrEnumerateDomainsInSamServer(dce, serverHandle)
            domains = resp['Buffer']['Buffer']
            logging.info('Looking up users in domain %s' % domains[0]['Name'])
            resp = samr.hSamrLookupDomainInSamServer(dce, serverHandle, domains[0]['Name'])
            resp = samr.hSamrOpenDomain(dce, serverHandle=serverHandle, domainId=resp['DomainId'])
            domainHandle = resp['DomainHandle']
            status = STATUS_MORE_ENTRIES
            enumerationContext = 0
            while status == STATUS_MORE_ENTRIES:
                try:
                    resp = samr.hSamrEnumerateUsersInDomain(dce, domainHandle, samr.USER_WORKSTATION_TRUST_ACCOUNT, enumerationContext=enumerationContext)
                except DCERPCException as e:
                    if str(e).find('STATUS_MORE_ENTRIES') < 0:
                        raise
                    resp = e.get_packet()
                for user in resp['Buffer']['Buffer']:
                    self.__machinesList.append(user['Name'][:-1])
                    logging.debug('Machine name - rid: %s - %d' % (user['Name'], user['RelativeId']))
                enumerationContext = resp['EnumerationContext']
                status = resp['ErrorCode']
        except Exception as e:
            raise e
        dce.disconnect()

    def getTargets(self):
        if False:
            return 10
        logging.info('Importing targets')
        if self.__options.target is None and self.__options.targets is None:
            self.getDomainMachines()
        elif self.__options.targets is not None:
            for line in self.__options.targets.readlines():
                self.__machinesList.append(line.strip(' \r\n'))
        else:
            self.__machinesList.append(self.__options.target)
        logging.info('Got %d machines' % len(self.__machinesList))

    def filterUsers(self):
        if False:
            print('Hello World!')
        if self.__options.user is not None:
            self.__filterUsers = list()
            self.__filterUsers.append(self.__options.user)
        elif self.__options.users is not None:
            self.__filterUsers = list()
            for line in self.__options.users.readlines():
                self.__filterUsers.append(line.strip(' \r\n'))
        else:
            self.__filterUsers = None

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.getTargets()
        self.filterUsers()
        self.__targetsThreadEvent = Event()
        if self.__options.noloop is False:
            self.__targetsThread = Thread(target=checkMachines, args=(self.__machinesList, self.__targetsThreadEvent))
            self.__targetsThread.start()
        else:
            checkMachines(self.__machinesList, self.__targetsThreadEvent, singlePass=True)
        while True:
            while machinesAliveQueue.empty() is False:
                machine = machinesAliveQueue.get()
                logging.debug('Adding %s to the up list' % machine)
                self.__targets[machine] = {}
                self.__targets[machine]['SRVS'] = None
                self.__targets[machine]['WKST'] = None
                self.__targets[machine]['Admin'] = True
                self.__targets[machine]['Sessions'] = list()
                self.__targets[machine]['LoggedIn'] = set()
            for target in list(self.__targets.keys()):
                try:
                    self.getSessions(target)
                    self.getLoggedIn(target)
                except (SessionError, DCERPCException) as e:
                    if str(e).find('LOGON_FAILURE') >= 0:
                        logging.error('STATUS_LOGON_FAILURE for %s, discarding' % target)
                        del self.__targets[target]
                    elif str(e).find('INVALID_PARAMETER') >= 0:
                        del self.__targets[target]
                    elif str(e).find('access_denied') >= 0:
                        del self.__targets[target]
                    else:
                        logging.info(str(e))
                    pass
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    if str(e).find('timed out') >= 0:
                        del self.__targets[target]
                        machinesDownQueue.put(target)
                    else:
                        logging.error(e)
                    pass
            if self.__options.noloop is True:
                break
            logging.debug('Sleeping for %s seconds' % self.__options.delay)
            logging.debug('Currently monitoring %d active targets' % len(self.__targets))
            sleep(int(self.__options.delay))

    def getSessions(self, target):
        if False:
            i = 10
            return i + 15
        if self.__targets[target]['SRVS'] is None:
            stringSrvsBinding = 'ncacn_np:%s[\\PIPE\\srvsvc]' % target
            rpctransportSrvs = transport.DCERPCTransportFactory(stringSrvsBinding)
            if hasattr(rpctransportSrvs, 'set_credentials'):
                rpctransportSrvs.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey)
                rpctransportSrvs.set_kerberos(self.__doKerberos, self.__kdcHost)
            dce = rpctransportSrvs.get_dce_rpc()
            dce.connect()
            dce.bind(srvs.MSRPC_UUID_SRVS)
            self.__maxConnections -= 1
        else:
            dce = self.__targets[target]['SRVS']
        try:
            resp = srvs.hNetrSessionEnum(dce, '\x00', NULL, 10)
        except Exception as e:
            if str(e).find('Broken pipe') >= 0:
                self.__targets[target]['SRVS'] = None
                self.__maxConnections += 1
                return
            else:
                raise
        if self.__maxConnections < 0:
            dce.disconnect()
            self.__maxConnections = 0
        else:
            self.__targets[target]['SRVS'] = dce
        tmpSession = list()
        printCRLF = False
        for session in resp['InfoStruct']['SessionInfo']['Level10']['Buffer']:
            userName = session['sesi10_username'][:-1]
            sourceIP = session['sesi10_cname'][:-1][2:]
            key = '%s\x01%s' % (userName, sourceIP)
            myEntry = '%s\x01%s' % (self.__username, myIP)
            tmpSession.append(key)
            if not key in self.__targets[target]['Sessions']:
                if key != myEntry:
                    self.__targets[target]['Sessions'].append(key)
                    if self.__filterUsers is not None:
                        if userName in self.__filterUsers:
                            print('%s: user %s logged from host %s - active: %d, idle: %d' % (target, userName, sourceIP, session['sesi10_time'], session['sesi10_idle_time']))
                            printCRLF = True
                    else:
                        print('%s: user %s logged from host %s - active: %d, idle: %d' % (target, userName, sourceIP, session['sesi10_time'], session['sesi10_idle_time']))
                        printCRLF = True
        for (nItem, session) in enumerate(self.__targets[target]['Sessions']):
            (userName, sourceIP) = session.split('\x01')
            if session not in tmpSession:
                del self.__targets[target]['Sessions'][nItem]
                if self.__filterUsers is not None:
                    if userName in self.__filterUsers:
                        print('%s: user %s logged off from host %s' % (target, userName, sourceIP))
                        printCRLF = True
                else:
                    print('%s: user %s logged off from host %s' % (target, userName, sourceIP))
                    printCRLF = True
        if printCRLF is True:
            print()

    def getLoggedIn(self, target):
        if False:
            i = 10
            return i + 15
        if self.__targets[target]['Admin'] is False:
            return
        if self.__targets[target]['WKST'] is None:
            stringWkstBinding = 'ncacn_np:%s[\\PIPE\\wkssvc]' % target
            rpctransportWkst = transport.DCERPCTransportFactory(stringWkstBinding)
            if hasattr(rpctransportWkst, 'set_credentials'):
                rpctransportWkst.set_credentials(self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, self.__aesKey)
                rpctransportWkst.set_kerberos(self.__doKerberos, self.__kdcHost)
            dce = rpctransportWkst.get_dce_rpc()
            dce.connect()
            dce.bind(wkst.MSRPC_UUID_WKST)
            self.__maxConnections -= 1
        else:
            dce = self.__targets[target]['WKST']
        try:
            resp = wkst.hNetrWkstaUserEnum(dce, 1)
        except Exception as e:
            if str(e).find('Broken pipe') >= 0:
                self.__targets[target]['WKST'] = None
                self.__maxConnections += 1
                return
            elif str(e).upper().find('ACCESS_DENIED'):
                dce.disconnect()
                self.__maxConnections += 1
                self.__targets[target]['Admin'] = False
                return
            else:
                raise
        if self.__maxConnections < 0:
            dce.disconnect()
            self.__maxConnections = 0
        else:
            self.__targets[target]['WKST'] = dce
        tmpLoggedUsers = set()
        printCRLF = False
        for session in resp['UserInfo']['WkstaUserInfo']['Level1']['Buffer']:
            userName = session['wkui1_username'][:-1]
            logonDomain = session['wkui1_logon_domain'][:-1]
            key = '%s\x01%s' % (userName, logonDomain)
            tmpLoggedUsers.add(key)
            if not key in self.__targets[target]['LoggedIn']:
                self.__targets[target]['LoggedIn'].add(key)
                if self.__filterUsers is not None:
                    if userName in self.__filterUsers:
                        print('%s: user %s\\%s logged in LOCALLY' % (target, logonDomain, userName))
                        printCRLF = True
                else:
                    print('%s: user %s\\%s logged in LOCALLY' % (target, logonDomain, userName))
                    printCRLF = True
        for session in self.__targets[target]['LoggedIn'].copy():
            (userName, logonDomain) = session.split('\x01')
            if session not in tmpLoggedUsers:
                self.__targets[target]['LoggedIn'].remove(session)
                if self.__filterUsers is not None:
                    if userName in self.__filterUsers:
                        print('%s: user %s\\%s logged off LOCALLY' % (target, logonDomain, userName))
                        printCRLF = True
                else:
                    print('%s: user %s\\%s logged off LOCALLY' % (target, logonDomain, userName))
                    printCRLF = True
        if printCRLF is True:
            print()

    def stop(self):
        if False:
            print('Hello World!')
        if self.__targetsThreadEvent is not None:
            self.__targetsThreadEvent.set()
if __name__ == '__main__':
    print(version.BANNER)
    parser = argparse.ArgumentParser()
    parser.add_argument('identity', action='store', help='[domain/]username[:password]')
    parser.add_argument('-user', action='store', help='Filter output by this user')
    parser.add_argument('-users', type=argparse.FileType('r'), help='input file with list of users to filter to output for')
    parser.add_argument('-target', action='store', help='target system to query info from. If not specified script will run in domain mode.')
    parser.add_argument('-targets', type=argparse.FileType('r'), help='input file with targets system to query info from (one per line). If not specified script will run in domain mode.')
    parser.add_argument('-noloop', action='store_true', default=False, help='Stop after the first probe')
    parser.add_argument('-delay', action='store', default='10', help='seconds delay between starting each batch probe (default 10 seconds)')
    parser.add_argument('-max-connections', action='store', default='1000', help='Max amount of connections to keep opened (default 1000)')
    parser.add_argument('-ts', action='store_true', help='Adds timestamp to every logging output')
    parser.add_argument('-debug', action='store_true', help='Turn DEBUG output ON')
    group = parser.add_argument_group('authentication')
    group.add_argument('-hashes', action='store', metavar='LMHASH:NTHASH', help='NTLM hashes, format is LMHASH:NTHASH')
    group.add_argument('-no-pass', action='store_true', help="don't ask for password (useful for -k)")
    group.add_argument('-k', action='store_true', help='Use Kerberos authentication. Grabs credentials from ccache file (KRB5CCNAME) based on target parameters. If valid credentials cannot be found, it will use the ones specified in the command line')
    group.add_argument('-aesKey', action='store', metavar='hex key', help='AES key to use for Kerberos Authentication (128 or 256 bits)')
    group.add_argument('-dc-ip', action='store', metavar='ip address', help='IP Address of the domain controller. If ommited it use the domain part (FQDN) specified in the target parameter')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    options = parser.parse_args()
    logger.init(options.ts)
    if options.debug is True:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug(version.getInstallationPath())
    else:
        logging.getLogger().setLevel(logging.INFO)
    (domain, username, password) = parse_credentials(options.identity)
    try:
        if domain is None:
            domain = ''
        if password == '' and username != '' and (options.hashes is None) and (options.no_pass is False) and (options.aesKey is None):
            from getpass import getpass
            password = getpass('Password:')
        if options.aesKey is not None:
            options.k = True
        executer = USERENUM(username, password, domain, options.hashes, options.aesKey, options.k, options)
        executer.run()
    except Exception as e:
        if logging.getLogger().level == logging.DEBUG:
            import traceback
            traceback.print_exc()
        logging.error(e)
        executer.stop()
    except KeyboardInterrupt:
        logging.info('Quitting.. please wait')
        executer.stop()
    sys.exit(0)