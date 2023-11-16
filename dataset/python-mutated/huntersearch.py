import asyncio
from typing import List
from theHarvester.discovery.constants import MissingKey
from theHarvester.lib.core import AsyncFetcher, Core

class SearchHunter:

    def __init__(self, word, limit, start) -> None:
        if False:
            while True:
                i = 10
        self.word = word
        self.limit = limit
        self.limit = 10 if limit > 10 else limit
        self.start = start
        self.key = Core.hunter_key()
        if self.key is None:
            raise MissingKey('Hunter')
        self.total_results = ''
        self.counter = start
        self.database = f'https://api.hunter.io/v2/domain-search?domain={self.word}&api_key={self.key}&limit=10'
        self.proxy = False
        self.hostnames: List = []
        self.emails: List = []

    async def do_search(self) -> None:
        is_free = True
        headers = {'User-Agent': Core.get_user_agent()}
        acc_info_url = f'https://api.hunter.io/v2/account?api_key={self.key}'
        response = await AsyncFetcher.fetch_all([acc_info_url], headers=headers, json=True)
        is_free = is_free if 'plan_name' in response[0]['data'].keys() and response[0]['data']['plan_name'].lower() == 'free' else False
        total_requests_avail = response[0]['data']['requests']['searches']['available'] - response[0]['data']['requests']['searches']['used']
        if is_free:
            response = await AsyncFetcher.fetch_all([self.database], headers=headers, proxy=self.proxy, json=True)
            (self.emails, self.hostnames) = await self.parse_resp(json_resp=response[0])
        else:
            hunter_dinfo_url = f'https://api.hunter.io/v2/email-count?domain={self.word}'
            response = await AsyncFetcher.fetch_all([hunter_dinfo_url], headers=headers, proxy=self.proxy, json=True)
            total_number_reqs = response[0]['data']['total'] // 100
            if total_requests_avail < total_number_reqs:
                print('WARNING: account does not have enough requests to gather all emails')
                print(f'Total requests available: {total_requests_avail}, total requests needed to be made: {total_number_reqs}')
                print('RETURNING current results, if you would still like to run this module comment out the if request')
                return
            self.limit = 100
            for offset in range(0, 100 * total_number_reqs, 100):
                req_url = f'https://api.hunter.io/v2/domain-search?domain={self.word}&api_key={self.key}&limit{self.limit}&offset={offset}'
                response = await AsyncFetcher.fetch_all([req_url], headers=headers, proxy=self.proxy, json=True)
                (temp_emails, temp_hostnames) = await self.parse_resp(response[0])
                self.emails.extend(temp_emails)
                self.hostnames.extend(temp_hostnames)
                await asyncio.sleep(1)

    async def parse_resp(self, json_resp):
        emails = list(sorted({email['value'] for email in json_resp['data']['emails']}))
        domains = list(sorted({source['domain'] for email in json_resp['data']['emails'] for source in email['sources'] if self.word in source['domain']}))
        return (emails, domains)

    async def process(self, proxy: bool=False) -> None:
        self.proxy = proxy
        await self.do_search()

    async def get_emails(self):
        return self.emails

    async def get_hostnames(self):
        return self.hostnames