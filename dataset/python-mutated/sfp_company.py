import re
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_company(SpiderFootPlugin):
    meta = {'name': 'Company Name Extractor', 'summary': 'Identify company names in any obtained data.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {'filterjscss': True}
    optdescs = {'filterjscss': 'Filter out company names that originated from CSS/JS content. Enabling this avoids detection of popular Javascript and web framework author company names.'}

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            while True:
                i = 10
        return ['TARGET_WEB_CONTENT', 'SSL_CERTIFICATE_ISSUED', 'DOMAIN_WHOIS', 'NETBLOCK_WHOIS', 'AFFILIATE_DOMAIN_WHOIS', 'AFFILIATE_WEB_CONTENT']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['COMPANY_NAME', 'AFFILIATE_COMPANY_NAME']

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        pattern_prefix = '(?=[,;:\\\'\\">\\(= ]|^)\\s?([A-Z0-9\\(\\)][A-Za-z0-9\\-&,\\.][^ \\"\\\';:><]*)?\\s?([A-Z0-9\\(\\)][A-Za-z0-9\\-&,\\.]?[^ \\"\\\';:><]*|[Aa]nd)?\\s?([A-Z0-9\\(\\)][A-Za-z0-9\\-&,\\.]?[^ \\"\\\';:><]*)?\\s+'
        pattern_match_re = ['LLC', 'L\\.L\\.C\\.?', 'AG', 'A\\.G\\.?', 'GmbH', 'Pty\\.?\\s+Ltd\\.?', 'Ltd\\.?', 'Pte\\.?', 'Inc\\.?', 'INC\\.?', 'Incorporated', 'Foundation', 'Corp\\.?', 'Corporation', 'SA', 'S\\.A\\.?', 'SIA', 'BV', 'B\\.V\\.?', 'NV', 'N\\.V\\.?', 'PLC', 'Limited', 'Pvt\\.?\\s+Ltd\\.?', 'SARL']
        pattern_match = ['LLC', 'L.L.C', 'AG', 'A.G', 'GmbH', 'Pty', 'Ltd', 'Pte', 'Inc', 'INC', 'Foundation', 'Corp', 'SA', 'S.A', 'SIA', 'BV', 'B.V', 'NV', 'N.V', 'PLC', 'Limited', 'Pvt.', 'SARL']
        pattern_suffix = '(?=[ \\.,:<\\)\\\'\\"]|[$\\n\\r])'
        filterpatterns = ['Copyright', '\\d{4}']
        if eventName in ['COMPANY_NAME', 'AFFILIATE_COMPANY_NAME']:
            return
        if eventName == 'TARGET_WEB_CONTENT':
            url = event.actualSource
            if self.opts['filterjscss'] and ('.js' in url or '.css' in url):
                self.debug('Ignoring web content from CSS/JS.')
                return
        self.debug(f'Received event, {eventName}, from {srcModuleName} ({len(eventData)} bytes)')
        try:
            if eventName == 'SSL_CERTIFICATE_ISSUED':
                eventData = eventData.split('O=')[1]
        except Exception:
            self.debug("Couldn't strip out 'O=' from certificate issuer, proceeding anyway...")
        chunks = list()
        for pat in pattern_match:
            start = 0
            m = eventData.find(pat, start)
            while m > 0:
                start = m - 50
                if start < 0:
                    start = 0
                end = m + 10
                if end >= len(eventData):
                    end = len(eventData) - 1
                chunks.append(eventData[start:end])
                offset = m + len(pat)
                m = eventData.find(pat, offset)
        myres = list()
        for chunk in chunks:
            for pat in pattern_match_re:
                matches = re.findall(pattern_prefix + '(' + pat + ')' + pattern_suffix, chunk, re.MULTILINE | re.DOTALL)
                for match in matches:
                    matched = 0
                    for m in match:
                        if len(m) > 0:
                            matched += 1
                    if matched <= 1:
                        continue
                    fullcompany = ''
                    for m in match:
                        flt = False
                        for f in filterpatterns:
                            if re.match(f, m):
                                flt = True
                        if not flt:
                            fullcompany += m + ' '
                    fullcompany = re.sub('\\s+', ' ', fullcompany.strip())
                    self.info('Found company name: ' + fullcompany)
                    if fullcompany in myres:
                        self.debug('Already found from this source.')
                        continue
                    myres.append(fullcompany)
                    if 'AFFILIATE_' in eventName:
                        etype = 'AFFILIATE_COMPANY_NAME'
                    else:
                        etype = 'COMPANY_NAME'
                    evt = SpiderFootEvent(etype, fullcompany, self.__name__, event)
                    if event.moduleDataSource:
                        evt.moduleDataSource = event.moduleDataSource
                    else:
                        evt.moduleDataSource = 'Unknown'
                    self.notifyListeners(evt)