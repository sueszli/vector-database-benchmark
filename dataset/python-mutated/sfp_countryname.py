import re
import phonenumbers
from phonenumbers.phonenumberutil import region_code_for_country_code
from spiderfoot import SpiderFootEvent, SpiderFootHelpers, SpiderFootPlugin

class sfp_countryname(SpiderFootPlugin):
    meta = {'name': 'Country Name Extractor', 'summary': 'Identify country names in any obtained data.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Content Analysis']}
    opts = {'cohosted': True, 'affiliate': True, 'noncountrytld': True, 'similardomain': False}
    optdescs = {'cohosted': 'Obtain country name from co-hosted sites', 'affiliate': 'Obtain country name from affiliate sites', 'noncountrytld': 'Parse TLDs not associated with any country as default country domains', 'similardomain': 'Obtain country name from similar domains'}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            return 10
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in userOpts.keys():
            self.opts[opt] = userOpts[opt]

    def detectCountryFromPhone(self, srcPhoneNumber: str) -> str:
        if False:
            return 10
        'Lookup name of country from phone number region code.\n\n        Args:\n            srcPhoneNumber (str): phone number\n\n        Returns:\n            str: country name\n        '
        if not isinstance(srcPhoneNumber, str):
            return None
        try:
            phoneNumber = phonenumbers.parse(srcPhoneNumber)
        except Exception:
            self.debug(f'Skipped invalid phone number: {srcPhoneNumber}')
            return None
        try:
            countryCode = region_code_for_country_code(phoneNumber.country_code)
        except Exception:
            self.debug(f'Lookup of region code failed for phone number: {srcPhoneNumber}')
            return None
        if not countryCode:
            return None
        return SpiderFootHelpers.countryNameFromCountryCode(countryCode.upper())

    def detectCountryFromDomainName(self, srcDomain: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Lookup name of country from TLD of domain name.\n\n        Args:\n            srcDomain (str): domain\n\n        Returns:\n            str: country name\n        '
        if not isinstance(srcDomain, str):
            return None
        domainParts = srcDomain.split('.')
        for part in domainParts[::-1]:
            country_name = SpiderFootHelpers.countryNameFromTld(part)
            if country_name:
                return country_name
        return None

    def detectCountryFromIBAN(self, srcIBAN: str) -> str:
        if False:
            return 10
        'Detect name of country from IBAN.\n\n        Args:\n            srcIBAN (str): IBAN\n\n        Returns:\n            str: country name\n        '
        if not isinstance(srcIBAN, str):
            return None
        return SpiderFootHelpers.countryNameFromCountryCode(srcIBAN[0:2])

    def detectCountryFromData(self, srcData: str) -> list:
        if False:
            i = 10
            return i + 15
        'Detect name of country from event data (WHOIS lookup, Geo Info, Physical Address, etc)\n\n        Args:\n            srcData (str): event data\n\n        Returns:\n            list: list of countries\n        '
        countries = list()
        if not srcData:
            return countries
        abbvCountryCodes = SpiderFootHelpers.countryCodes()
        for countryName in abbvCountryCodes.values():
            if countryName.lower() not in srcData.lower():
                continue
            matchCountries = re.findall('[,\'\\"\\:\\=\\[\\(\\[\\n\\t\\r\\.] ?' + countryName + '[,\'\\"\\:\\=\\[\\(\\[\\n\\t\\r\\.]', srcData, re.IGNORECASE)
            if matchCountries:
                countries.append(countryName)
        matchCountries = re.findall('country: (.+?)', srcData, re.IGNORECASE)
        if matchCountries:
            for m in matchCountries:
                m = m.strip()
                if m in abbvCountryCodes:
                    countries.append(abbvCountryCodes[m])
                if m in abbvCountryCodes.values():
                    countries.append(m)
        return list(set(countries))

    def watchedEvents(self):
        if False:
            return 10
        return ['IBAN_NUMBER', 'PHONE_NUMBER', 'AFFILIATE_DOMAIN_NAME', 'CO_HOSTED_SITE_DOMAIN', 'DOMAIN_NAME', 'SIMILARDOMAIN', 'AFFILIATE_DOMAIN_WHOIS', 'CO_HOSTED_SITE_DOMAIN_WHOIS', 'DOMAIN_WHOIS', 'GEOINFO', 'PHYSICAL_ADDRESS']

    def producedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['COUNTRY_NAME']

    def handleEvent(self, event):
        if False:
            while True:
                i = 10
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        if event.moduleDataSource:
            moduleDataSource = event.moduleDataSource
        else:
            moduleDataSource = 'Unknown'
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        eventDataHash = self.sf.hashstring(eventData)
        if eventDataHash in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventDataHash] = True
        countryNames = list()
        if eventName == 'PHONE_NUMBER':
            countryNames.append(self.detectCountryFromPhone(eventData))
        elif eventName == 'DOMAIN_NAME':
            countryNames.append(self.detectCountryFromDomainName(eventData))
        elif eventName == 'AFFILIATE_DOMAIN_NAME' and self.opts['affiliate']:
            countryNames.append(self.detectCountryFromDomainName(eventData))
        elif eventName == 'CO_HOSTED_SITE_DOMAIN' and self.opts['cohosted']:
            countryNames.append(self.detectCountryFromDomainName(eventData))
        elif eventName == 'SIMILARDOMAIN' and self.opts['similardomain']:
            countryNames.append(self.detectCountryFromDomainName(eventData))
        elif eventName == 'IBAN_NUMBER':
            countryNames.append(self.detectCountryFromIBAN(eventData))
        elif eventName in ['DOMAIN_WHOIS', 'GEOINFO', 'PHYSICAL_ADDRESS']:
            countryNames.extend(self.detectCountryFromData(eventData))
        elif eventName == 'AFFILIATE_DOMAIN_WHOIS' and self.opts['affiliate']:
            countryNames.extend(self.detectCountryFromData(eventData))
        elif eventName == 'CO_HOSTED_SITE_DOMAIN_WHOIS' and self.opts['cohosted']:
            countryNames.extend(self.detectCountryFromData(eventData))
        if not countryNames:
            self.debug(f'Found no country names associated with {eventName}: {eventData}')
            return
        for countryName in set(countryNames):
            if not countryName:
                continue
            self.debug(f'Found country name: {countryName}')
            evt = SpiderFootEvent('COUNTRY_NAME', countryName, self.__name__, event)
            evt.moduleDataSource = moduleDataSource
            self.notifyListeners(evt)