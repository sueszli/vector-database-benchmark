import json
import time
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_etherscan(SpiderFootPlugin):
    meta = {'name': 'Etherscan', 'summary': 'Queries etherscan.io to find the balance of identified ethereum wallet addresses.', 'flags': ['apikey'], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Public Registries'], 'dataSource': {'website': 'https://etherscan.io', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://etherscan.io/apis'], 'apiKeyInstructions': ['Visit https://etherscan.io', 'Register a free account', 'Browse to https://etherscan.io/myapikey', 'Click on Add beside API Key', 'Your API Key will be listed under API Key Token'], 'favIcon': 'https://etherscan.io/images/favicon3.ico', 'logo': 'https://etherscan.io/images/brandassets/etherscan-logo-circle.png', 'description': 'Etherscan allows you to explore and search the Ethereum blockchain for transactions, addresses, tokens, prices and other activities taking place on Ethereum (ETH)'}}
    opts = {'api_key': '', 'pause': 1}
    optdescs = {'api_key': 'API Key for etherscan.io', 'pause': 'Number of seconds to wait between each API call.'}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            while True:
                i = 10
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            i = 10
            return i + 15
        return ['ETHEREUM_ADDRESS']

    def producedEvents(self):
        if False:
            return 10
        return ['ETHEREUM_BALANCE', 'RAW_RIR_DATA']

    def query(self, qry):
        if False:
            while True:
                i = 10
        queryString = f"https://api.etherscan.io/api?module=account&action=balance&address={qry}&tag=latest&apikey={self.opts['api_key']}"
        res = self.sf.fetchUrl(queryString, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        time.sleep(self.opts['pause'])
        if res['content'] is None:
            self.info(f'No Etherscan data found for {qry}')
            return None
        try:
            return json.loads(res['content'])
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
        return None

    def handleEvent(self, event):
        if False:
            print('Hello World!')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if self.errorState:
            return
        if self.opts['api_key'] == '':
            self.error('You enabled sfp_etherscan but did not set an API key!')
            self.errorState = True
            return
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        data = self.query(eventData)
        if data is None:
            self.info(f'No Etherscan data found for {eventData}')
            return
        balance = float(data.get('result')) / 1000000000000000000
        evt = SpiderFootEvent('ETHEREUM_BALANCE', f'{str(balance)} ETH', self.__name__, event)
        self.notifyListeners(evt)
        evt = SpiderFootEvent('RAW_RIR_DATA', str(data), self.__name__, event)
        self.notifyListeners(evt)