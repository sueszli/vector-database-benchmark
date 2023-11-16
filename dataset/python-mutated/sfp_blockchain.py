import json
from spiderfoot import SpiderFootEvent, SpiderFootPlugin

class sfp_blockchain(SpiderFootPlugin):
    meta = {'name': 'Blockchain', 'summary': 'Queries blockchain.info to find the balance of identified bitcoin wallet addresses.', 'flags': [], 'useCases': ['Footprint', 'Investigate', 'Passive'], 'categories': ['Public Registries'], 'dataSource': {'website': 'https://www.blockchain.com/', 'model': 'FREE_NOAUTH_UNLIMITED', 'references': ['https://exchange.blockchain.com/api/#introduction', 'https://exchange.blockchain.com/markets', 'https://exchange.blockchain.com/fees', 'https://exchange.blockchain.com/trade'], 'favIcon': 'https://www.blockchain.com/static/favicon.ico', 'logo': 'https://exchange.blockchain.com/api/assets/images/logo.png', 'description': 'Blockchain Exchange is the most secure place to buy, sell, and trade crypto.\nUse the most popular block explorer to search and verify transactions on the Bitcoin, Ethereum, and Bitcoin Cash blockchains.\nStay on top of Bitcoin and other top cryptocurrency prices, news, and market information.'}}
    opts = {}
    optdescs = {}
    results = None

    def setup(self, sfc, userOpts=dict()):
        if False:
            i = 10
            return i + 15
        self.sf = sfc
        self.results = self.tempStorage()
        for opt in list(userOpts.keys()):
            self.opts[opt] = userOpts[opt]

    def watchedEvents(self):
        if False:
            print('Hello World!')
        return ['BITCOIN_ADDRESS']

    def producedEvents(self):
        if False:
            return 10
        return ['BITCOIN_BALANCE']

    def handleEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        eventName = event.eventType
        srcModuleName = event.module
        eventData = event.data
        self.debug(f'Received event, {eventName}, from {srcModuleName}')
        if eventData in self.results:
            self.debug(f'Skipping {eventData}, already checked.')
            return
        self.results[eventData] = True
        res = self.sf.fetchUrl('https://blockchain.info/balance?active=' + eventData, timeout=self.opts['_fetchtimeout'], useragent=self.opts['_useragent'])
        if res['content'] is None:
            self.info('No Blockchain info found for ' + eventData)
            return
        try:
            data = json.loads(res['content'])
            balance = float(data[eventData]['final_balance']) / 100000000
        except Exception as e:
            self.debug(f'Error processing JSON response: {e}')
            return
        evt = SpiderFootEvent('BITCOIN_BALANCE', str(balance) + ' BTC', self.__name__, event)
        self.notifyListeners(evt)