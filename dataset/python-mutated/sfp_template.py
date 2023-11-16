import json
from netaddr import IPNetwork
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_template(SpiderFootPlugin):
    meta = {'name': 'Template Module', 'summary': 'This is an example module to help developers create their own SpiderFoot modules.', 'flags': ['slow', 'apikey'], 'useCases': ['Passive'], 'categories': ['Social Media'], 'toolDetails': {'name': 'Nmap', 'description': 'Detailed descriptive text about the tool', 'website': 'https://tool.org', 'repository': 'https://github.com/author/tool'}, 'dataSource': {'website': 'https://www.datasource.com', 'model': 'FREE_NOAUTH_LIMITED', 'references': ['https://www.datasource.com/api-documentation'], 'apiKeyInstructions': ['Visit https://www.datasource.com', 'Register a free account', "Click on 'Account Settings'", "Click on 'Developer'", "The API key is listed under 'Your API Key'"], 'favIcon': 'https://www.datasource.com/favicon.ico', 'logo': 'https://www.datasource.com/logo.gif', 'description': 'A paragraph of text with details about the data source / services. Keep things neat by breaking the text up across multiple lines as has been done here. If line breaks are needed for breaking up multiple paragraphs, use \n.'}}
    opts = {'api_key': '', 'checkcohosts': True, 'checkaffiliates': True, 'subnetlookup': False, 'netblocklookup': True, 'maxsubnet': 24, 'maxnetblock': 24, 'maxcohost': 100, 'verify': True, 'cohostsamedomain': False}
    optdescs = {'api_key': 'SomeDataource API Key.', 'checkcohosts': 'Check co-hosted sites?', 'checkaffiliates': 'Check affiliates?', 'netblocklookup': 'Look up all IPs on netblocks deemed to be owned by your target for possible blacklisted hosts on the same target subdomain/domain?', 'maxnetblock': 'If looking up owned netblocks, the maximum netblock size to look up all IPs within (CIDR value, 24 = /24, 16 = /16, etc.)', 'subnetlookup': 'Look up all IPs on subnets which your target is a part of?', 'maxsubnet': 'If looking up subnets, the maximum subnet size to look up all the IPs within (CIDR value, 24 = /24, 16 = /16, etc.)', 'maxcohost': 'Stop reporting co-hosted sites after this many are found, as it would likely indicate web hosting.', 'cohostsamedomain': 'Treat co-hosted sites on the same target domain as co-hosting?', 'verify': 'Verify that any hostnames found on the target domain still resolve?'}
    results = None
    errorState = False

    def setup(self, sfc, userOpts=dict()):
        if False:
            print('Hello World!')
        self.sf = sfc
        self.results = self.tempStorage()
        self.__dataSource__ = 'Some Data Source'
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['IP_ADDRESS', 'NETBLOCK_OWNER', 'DOMAIN_NAME', 'WEB_ANALYTICS_ID']

    def producedEvents(self):
        if False:
            return 10
        return ['OPERATING_SYSTEM', 'DEVICE_TYPE', 'TCP_PORT_OPEN', 'TCP_PORT_OPEN_BANNER', 'RAW_RIR_DATA', 'GEOINFO', 'VULNERABILITY_GENERAL']

    def query(self, qry):
        if False:
            for i in range(10):
                print('nop')
        res = self.sf.fetchUrl(f"https://api.shodan.io/shodan/host/{qry}?key={self.opts['api_key']}", timeout=self.opts['_fetchtimeout'], useragent='SpiderFoot')
        if res['content'] is None:
            self.info(f'No SHODAN info found for {qry}')
            return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.error(f'Error processing JSON response from SHODAN: {e}')
        return None

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        eventData = event.data
        if self.errorState:
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        if eventName == 'NETBLOCK_OWNER':
            if not self.opts['netblocklookup']:
                return
            max_netblock = self.opts['maxnetblock']
            net_size = IPNetwork(eventData).prefixlen
            if net_size < max_netblock:
                self.debug(f'Network size {net_size} bigger than permitted: {max_netblock}')
                return
        qrylist = list()
        if eventName.startswith('NETBLOCK_'):
            for ipaddr in IPNetwork(eventData):
                qrylist.append(str(ipaddr))
                self.results[str(ipaddr)] = True
        else:
            qrylist.append(eventData)
        for addr in qrylist:
            rec = self.query(addr)
            if rec is None:
                continue
            if eventName == 'NETBLOCK_OWNER':
                pevent = SpiderFootEvent('IP_ADDRESS', addr, self.__name__, event)
                self.notifyListeners(pevent)
            elif eventName == 'NETBLOCK_MEMBER':
                pevent = SpiderFootEvent('AFFILIATE_IPADDR', addr, self.__name__, event)
                self.notifyListeners(pevent)
            else:
                pevent = event
            evt = SpiderFootEvent('RAW_RIR_DATA', str(rec), self.__name__, pevent)
            self.notifyListeners(evt)
            if self.checkForStop():
                return
            if event.moduleDataSource:
                evt.moduleDataSource = event.moduleDataSource
            else:
                evt.moduleDataSource = 'Unknown'
            os = rec.get('os')
            if os:
                evt = SpiderFootEvent('OPERATING_SYSTEM', f'{os} ({addr})', self.__name__, pevent)
                self.notifyListeners(evt)