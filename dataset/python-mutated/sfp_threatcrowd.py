import json
from netaddr import IPNetwork
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_threatcrowd(SpiderFootPlugin):
    meta = {'name': 'ThreatCrowd', 'summary': 'Obtain information from ThreatCrowd about identified IP addresses, domains and e-mail addresses.', 'flags': [], 'useCases': ['Investigate', 'Passive'], 'categories': ['Reputation Systems'], 'dataSource': {'website': 'https://www.threatcrowd.org', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://threatcrowd.blogspot.com/2015/03/tutorial.html'], 'favIcon': 'https://www.threatcrowd.org/img/favicon-32x32.png', 'logo': 'https://www.threatcrowd.org/img/home.png', 'description': 'The ThreatCrowd API allows you to quickly identify related infrastructure and malware.\nWith the ThreatCrowd API you can search for Domains, IP Addreses, E-mail adddresses, Filehashes, Antivirus detections.'}}
    opts = {'checkcohosts': True, 'checkaffiliates': True, 'netblocklookup': True, 'maxnetblock': 24, 'subnetlookup': True, 'maxsubnet': 24}
    optdescs = {'checkcohosts': 'Check co-hosted sites?', 'checkaffiliates': 'Check affiliates?', 'netblocklookup': 'Look up all IPs on netblocks deemed to be owned by your target for possible hosts on the same target subdomain/domain?', 'maxnetblock': 'If looking up owned netblocks, the maximum netblock size to look up all IPs within (CIDR value, 24 = /24, 16 = /16, etc.)', 'subnetlookup': 'Look up all IPs on subnets which your target is a part of?', 'maxsubnet': 'If looking up subnets, the maximum subnet size to look up all the IPs within (CIDR value, 24 = /24, 16 = /16, etc.)'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.errorState = False
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['IP_ADDRESS', 'AFFILIATE_IPADDR', 'INTERNET_NAME', 'CO_HOSTED_SITE', 'NETBLOCK_OWNER', 'EMAILADDR', 'NETBLOCK_MEMBER', 'AFFILIATE_INTERNET_NAME']

    def producedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['MALICIOUS_IPADDR', 'MALICIOUS_INTERNET_NAME', 'MALICIOUS_COHOST', 'MALICIOUS_AFFILIATE_INTERNET_NAME', 'MALICIOUS_AFFILIATE_IPADDR', 'MALICIOUS_NETBLOCK', 'MALICIOUS_SUBNET', 'MALICIOUS_EMAILADDR']

    def query(self, qry):
        if False:
            i = 10
            return i + 15
        url = None
        if self.sf.validIP(qry):
            url = 'https://www.threatcrowd.org/searchApi/v2/ip/report/?ip=' + qry
        if '@' in qry:
            url = 'https://www.threatcrowd.org/searchApi/v2/email/report/?email=' + qry
        if not url:
            url = 'https://www.threatcrowd.org/searchApi/v2/domain/report/?domain=' + qry
        res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent='SpiderFoot')
        if res['content'] is None:
            self.info(f'No ThreatCrowd info found for {qry}')
            return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.error(f'Error processing JSON response from ThreatCrowd: {e}')
            self.errorState = True
        return None

    def handleEvent(self, event):
        if False:
            return 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if self.errorState:
            return
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        if eventName.startswith('AFFILIATE') and (not self.opts['checkaffiliates']):
            return
        if eventName == 'CO_HOSTED_SITE' and (not self.opts['checkcohosts']):
            return
        if eventName == 'NETBLOCK_OWNER':
            if not self.opts['netblocklookup']:
                return
            max_netblock = self.opts['maxnetblock']
            if IPNetwork(eventData).prefixlen < max_netblock:
                self.debug(f'Network size bigger than permitted: {IPNetwork(eventData).prefixlen} > {max_netblock}')
                return
        if eventName == 'NETBLOCK_MEMBER':
            if not self.opts['subnetlookup']:
                return
            max_subnet = self.opts['maxsubnet']
            if IPNetwork(eventData).prefixlen < max_subnet:
                self.debug(f'Network size bigger than permitted: {IPNetwork(eventData).prefixlen} > {max_subnet}')
                return
        qrylist = list()
        if eventName.startswith('NETBLOCK_'):
            for ipaddr in IPNetwork(eventData):
                qrylist.append(str(ipaddr))
                self.results[str(ipaddr)] = True
        else:
            qrylist.append(eventData)
        for addr in qrylist:
            if self.checkForStop():
                return
            info = self.query(addr)
            if info is None:
                continue
            if info.get('votes', 0) < 0:
                self.info('Found ThreatCrowd URL data for ' + addr)
                if eventName in ['IP_ADDRESS'] or eventName.startswith('NETBLOCK_'):
                    evt = 'MALICIOUS_IPADDR'
                if eventName == 'AFFILIATE_IPADDR':
                    evt = 'MALICIOUS_AFFILIATE_IPADDR'
                if eventName == 'INTERNET_NAME':
                    evt = 'MALICIOUS_INTERNET_NAME'
                if eventName == 'AFFILIATE_INTERNET_NAME':
                    evt = 'MALICIOUS_AFFILIATE_INTERNET_NAME'
                if eventName == 'CO_HOSTED_SITE':
                    evt = 'MALICIOUS_COHOST'
                if eventName == 'EMAILADDR':
                    evt = 'MALICIOUS_EMAILADDR'
                infourl = '<SFURL>' + info.get('permalink') + '</SFURL>'
                e = SpiderFootEvent(evt, 'ThreatCrowd [' + addr + ']\n' + infourl, self.__name__, event)
                self.notifyListeners(e)