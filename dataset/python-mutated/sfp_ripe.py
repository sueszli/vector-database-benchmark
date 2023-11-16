import json
import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_ripe(SpiderFootPlugin):
    meta = {'name': 'RIPE', 'summary': 'Queries the RIPE registry (includes ARIN data) to identify netblocks and other info.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Public Registries'], 'dataSource': {'website': 'https://www.ripe.net/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://www.ripe.net/publications/ipv6-info-centre/training-and-materials', 'https://www.ripe.net/publications/ipv6-info-centre/ipv6-documents', 'https://www.ripe.net/manage-ips-and-asns/db/support/documentation/ripe-database-documentation', 'https://www.ripe.net/manage-ips-and-asns/db/support/documentation/ripe-database-documentation/updating-objects-in-the-ripe-database/6-1-restful-api'], 'favIcon': 'https://www.ripe.net/favicon.ico', 'logo': 'https://www.ripe.net/++resource++ripe.plonetheme.images/RIPE_NCC_logo.png', 'description': "We're an independent, not-for-profit membership organisation that supports the infrastructure of the Internet through technical coordination in our service region. Our most prominent activity is to act as the Regional Internet Registry (RIR) providing global Internet resources and related services (IPv4, IPv6 and AS Number resources) to members in our service region."}}
    opts = {}
    optdescs = {}
    results = None
    currentEventSrc = None
    memCache = None
    nbreported = None
    keywords = None
    lastContent = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        self.memCache = self.tempStorage()
        self.currentEventSrc = None
        self.nbreported = self.tempStorage()
        self.lastContent = None
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            for i in range(10):
                print('nop')
        return ['IP_ADDRESS', 'IPV6_ADDRESS', 'NETBLOCK_MEMBER', 'NETBLOCK_OWNER', 'NETBLOCKV6_MEMBER', 'NETBLOCKV6_OWNER', 'BGP_AS_OWNER', 'BGP_AS_MEMBER']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['NETBLOCK_MEMBER', 'NETBLOCK_OWNER', 'NETBLOCKV6_MEMBER', 'NETBLOCKV6_OWNER', 'BGP_AS_MEMBER', 'BGP_AS_OWNER', 'RAW_RIR_DATA']

    def fetchRir(self, url):
        if False:
            while True:
                i = 10
        if url in self.memCache:
            return self.memCache[url]
        res = self.sf.fetchUrl(url, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        if res['content'] is not None:
            self.memCache[url] = res
            self.lastContent = res['content']
        return res

    def ipNetblock(self, ipaddr):
        if False:
            for i in range(10):
                print('nop')
        prefix = None
        res = self.fetchRir(f'https://stat.ripe.net/data/network-info/data.json?resource={ipaddr}')
        if res['content'] is None:
            self.debug(f'No netblock info found/available for {ipaddr} at RIPE.')
            return None
        try:
            j = json.loads(res['content'])
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return None
        prefix = j['data'].get('prefix')
        if prefix is None:
            self.debug('Could not identify network prefix.')
            return None
        return prefix

    def queryWhois(self, qry):
        if False:
            i = 10
            return i + 15
        res = self.fetchRir(f'https://stat.ripe.net/data/whois/data.json?resource={qry}')
        if res['content'] is None:
            self.debug(f'No results for {qry} at RIPE.')
            return None
        try:
            data = json.loads(res['content'])
            return data.get('data')
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
        return None

    def netblockAs(self, prefix):
        if False:
            return 10
        whois = self.queryWhois(prefix)
        if not whois:
            return None
        try:
            if len(whois['irr_records']) > 0:
                data = whois['irr_records'][0]
            else:
                data = whois['records'][0]
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return None
        asn = None
        for rec in data:
            if rec['key'] == 'origin':
                asn = rec['value']
                break
        if not asn:
            return None
        return str(asn)

    def entityOwnerInfo(self, entity):
        if False:
            return 10
        whois = self.queryWhois(entity)
        if not whois:
            return None
        try:
            data = whois['records']
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return None
        ownerinfo = dict()
        ownerkeys = ['as', 'value', 'auth', 'desc', 'org', 'mnt', 'admin', 'tech']
        for rec in data:
            for d in rec:
                for k in ownerkeys:
                    key = d['key']
                    value = d['value']
                    if not key.lower().startswith(k):
                        continue
                    if value.lower() in ['null', 'none', 'none specified']:
                        continue
                    if key in ownerinfo:
                        ownerinfo[key].append(value)
                    else:
                        ownerinfo[key] = [value]
        self.debug(f'Found owner info: {ownerinfo}')
        return ownerinfo

    def asNetblocks(self, asn):
        if False:
            print('Hello World!')
        res = self.fetchRir(f'https://stat.ripe.net/data/announced-prefixes/data.json?resource=AS{asn}')
        if res['content'] is None:
            self.debug(f'No netblocks info found/available for AS{asn} at RIPE.')
            return None
        try:
            j = json.loads(res['content'])
            data = j['data']['prefixes']
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return None
        netblocks = list()
        for rec in data:
            prefix = rec['prefix']
            netblocks.append(rec['prefix'])
            self.info(f'Additional netblock found from same AS: {prefix}')
        return netblocks

    def asNeighbours(self, asn):
        if False:
            print('Hello World!')
        res = self.fetchRir(f'https://stat.ripe.net/data/asn-neighbours/data.json?resource=AS{asn}')
        if res['content'] is None:
            self.debug(f'No neighbour info found/available for AS{asn} at RIPE.')
            return None
        try:
            j = json.loads(res['content'])
            data = j['data']['neighbours']
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return None
        neighbours = list()
        for rec in data:
            neighbours.append(str(rec['asn']))
        return neighbours

    def findName(self, string):
        if False:
            for i in range(10):
                print('nop')
        for n in self.getTarget().getNames():
            if n in string:
                return True
        if self.keywords is None:
            self.keywords = self.sf.domainKeywords(self.getTarget().getNames(), self.opts['_internettlds'])
        rx = ['^{0}[-_/\\\'\\"\\\\\\.,\\?\\!\\s\\d].*', '.*[-_/\\\'\\"\\\\\\.,\\?\\!\\s]{0}$', '.*[-_/\\\'\\"\\\\\\.,\\?\\!\\s]{0}[-_/\\\'\\"\\\\\\.,\\?\\!\\s\\d].*']
        keywordList = set()
        for kw in self.keywords:
            keywordList.add(kw)
            keywordList.add(kw.replace('-', ' '))
            keywordList.add(kw.replace('-', '_'))
            keywordList.add(kw.replace('-', ''))
        for kw in keywordList:
            for r in rx:
                if re.match(r.format(kw), string, re.IGNORECASE) is not None:
                    return True
        return False

    def ownsAs(self, asn):
        if False:
            print('Hello World!')
        ownerinfo = self.entityOwnerInfo(asn)
        if not ownerinfo:
            return False
        for k in list(ownerinfo.keys()):
            items = ownerinfo[k]
            for item in items:
                if self.findName(item.lower()):
                    return True
        return False

    def ownsNetblock(self, netblock):
        if False:
            return 10
        ownerinfo = self.entityOwnerInfo(netblock)
        if not ownerinfo:
            return False
        for k in list(ownerinfo.keys()):
            items = ownerinfo[k]
            for item in items:
                if self.findName(item.lower()):
                    return True
        return False

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.currentEventSrc = event
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        if eventName == 'BGP_AS_OWNER':
            if eventData in self.nbreported:
                return
            self.nbreported[eventData] = True
            netblocks = self.asNetblocks(eventData)
            if not netblocks:
                return
            for netblock in netblocks:
                if netblock in self.results:
                    continue
                if ':' in netblock:
                    evt = SpiderFootEvent('NETBLOCKV6_OWNER', netblock, self.__name__, event)
                    self.notifyListeners(evt)
                else:
                    evt = SpiderFootEvent('NETBLOCK_OWNER', netblock, self.__name__, event)
                    self.notifyListeners(evt)
            evt = SpiderFootEvent('RAW_RIR_DATA', self.lastContent, self.__name__, event)
            self.notifyListeners(evt)
            return
        if eventName.startswith('NETBLOCK'):
            asn = self.netblockAs(eventData)
            if asn is None:
                self.debug(f'Could not identify BGP AS for {eventData}')
                return
            if eventName in ['NETBLOCK_OWNER', 'NETBLOCKV6_OWNER'] and self.ownsAs(asn):
                asevt = SpiderFootEvent('BGP_AS_OWNER', asn, self.__name__, event)
                self.notifyListeners(asevt)
                evt = SpiderFootEvent('RAW_RIR_DATA', self.lastContent, self.__name__, event)
                self.notifyListeners(evt)
            else:
                asevt = SpiderFootEvent('BGP_AS_MEMBER', asn, self.__name__, event)
                self.notifyListeners(asevt)
            return
        if eventName in ['IP_ADDRESS', 'IPV6_ADDRESS']:
            prefix = self.ipNetblock(eventData)
            if prefix is None:
                self.debug(f'Could not identify network prefix for {eventData}')
                return
            asn = self.netblockAs(prefix)
            if asn is None:
                self.debug(f'Could not identify BGP AS for {prefix}')
                return
            if not self.sf.validIpNetwork(prefix):
                return
            self.info(f'Netblock found: {prefix} ({asn})')
            if self.ownsNetblock(prefix):
                relationship = 'OWNER'
            else:
                relationship = 'MEMBER'
            if ':' in prefix:
                evt = SpiderFootEvent('NETBLOCKV6_' + relationship, prefix, self.__name__, event)
            else:
                evt = SpiderFootEvent('NETBLOCK_' + relationship, prefix, self.__name__, event)
            self.notifyListeners(evt)