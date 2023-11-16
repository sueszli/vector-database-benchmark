import requests
import re
from bs4 import BeautifulSoup
from datetime import date
from os.path import exists, join
from functions import write_json, read_json
from models.bill import Bill
from config import BASE_URL, CACHE_BASE_PATH, OUTPUT_BASE_PATH
BILL_LIST_HTML_CACHE_PATH = join(CACHE_BASE_PATH, 'all-introduced-bills.html')
BILL_DATA_CACHE = join(CACHE_BASE_PATH, 'last-scrape-bill-data.json')
LAST_SCRAPE_BILLS_PATH = join(OUTPUT_BASE_PATH, 'all-bills.json')
TODAY = date.today().strftime('%m/%d/%Y')

class BillList:
    """Data structure for gathering list of bills from LAWS system

    Cache Logic: Import bills where either bill page "Status Date" or "Status" don't match cached version
        OR where the date of the last status update is today.

    """

    def __init__(self, bill_list_url, force_refresh=False, use_html_bill_list_cache=False, use_verbose_logging=False):
        if False:
            return 10
        bill_list = self.get_bill_list(bill_list_url, use_cache=use_html_bill_list_cache)
        self.use_verbose_logging = use_verbose_logging
        if exists(BILL_DATA_CACHE):
            self.last_scrape_bills = read_json(BILL_DATA_CACHE)
        else:
            if use_verbose_logging:
                print('No bill data cache found at', BILL_DATA_CACHE)
            self.last_scrape_bills = []
        self.bills = []
        for raw in bill_list:
            matches = [last for last in self.last_scrape_bills if last['key'] == raw['key']]
            needs_refresh = force_refresh or len(matches) == 0 or raw['statusDate'] != matches[0]['statusDate'] or (raw['lastAction'] != matches[0]['lastAction']) or (raw['statusDate'] == TODAY)
            bill = Bill(raw, needs_refresh=needs_refresh, use_verbose_logging=self.use_verbose_logging)
            self.bills.append(bill)
        self.write_bill_data_cache()

    def get_bill_list(self, list_url, use_cache=False, write_cache=True):
        if False:
            i = 10
            return i + 15
        if use_cache:
            print('Reading bill list from', BILL_LIST_HTML_CACHE_PATH)
            with open(BILL_LIST_HTML_CACHE_PATH, 'r') as f:
                text = f.read()
                parsed = self.parse_bill_list_html(text)
                return parsed
        else:
            print('Fetching bill list from', list_url)
            r = requests.get(list_url)
            text = r.text
            if write_cache:
                print('Writing bill list to', BILL_LIST_HTML_CACHE_PATH)
                with open(BILL_LIST_HTML_CACHE_PATH, 'w') as f:
                    f.write(text)
            parsed = self.parse_bill_list_html(text)
            return parsed

    def parse_bill_list_html(self, text):
        if False:
            i = 10
            return i + 15
        '\n        Returns bill list as raw dicts\n        '
        TEXT_BEFORE_TABLE = 'Total number of Introduced and Unintroduced Bills'
        soup = BeautifulSoup(text, 'html.parser')
        table_title = soup.find(text=re.compile(TEXT_BEFORE_TABLE))
        bill_table = table_title.find_next_sibling('table')
        rows = bill_table.find_all('tr')
        headers = [th.text for th in rows[0].find_all('th')]
        bills = [self.parse_bill_row(node, headers) for node in rows[1:]]
        return bills

    def parse_bill_row(self, node, keys):
        if False:
            while True:
                i = 10
        cells = [td.text for td in node.find_all('td')]
        links = node.find_all('a', href=True)
        bill_page_link = links[0]['href']
        bill_html_link = links[1]['href']
        bill_pdf_link = links[2]['href']
        raw = {}
        for (i, key) in enumerate(keys):
            raw[key] = cells[i]
        sponsor_raw = raw['Primary Sponsor'].replace('|', '')
        if 'Party/District Not Assigned' in sponsor_raw:
            print(f'Bill list bug; "{sponsor_raw}"')
        print(sponsor_raw)
        if sponsor_raw == 'Sara  Hess\xa0Party/District Not Assigned':
            sponsor_district = 'HD 69'
            sponsor_party = 'R'
            sponsor_name = 'Jennifer Carlson'
        else:
            sponsor_district = re.search('(H|S)D \\d+', sponsor_raw).group()
            sponsor_party = re.search('R|D(?=\\) (H|S)D \\d+)', sponsor_raw).group()
            sponsor_name = re.search('.+(?=\\(R|D\\) (H|S)D \\d+)', sponsor_raw).group().strip().replace('  ', ' ')
            sponsor_name = re.sub('\\($', '', sponsor_name).strip()
        bill = {'key': raw['Bill Type - Number'].replace('\xa0', ''), 'billPageUrl': ''.join([BASE_URL, bill_page_link]), 'billTextUrl': bill_html_link, 'billPdfUrl': bill_pdf_link, 'lc': raw['LC Number'], 'title': raw['Short Title'], 'sponsor': sponsor_name, 'sponsorParty': sponsor_party, 'sponsorDistrict': sponsor_district, 'statusDate': raw['Status Date'], 'lastAction': raw['Status'].replace('|', '')}
        return bill

    def write_bill_data_cache(self):
        if False:
            print('Hello World!')
        bill_list = []
        for bill in self.bills:
            bill_data = bill.export()
            bill_list.append(bill_data)
        write_json(bill_list, BILL_DATA_CACHE)

    def export(self):
        if False:
            while True:
                i = 10
        bill_list = []
        action_list = []
        vote_list = []
        for bill in self.bills:
            bill_data = bill.export()
            write_json(bill_data, join(OUTPUT_BASE_PATH, f'{bill.urlKey}--data.json'), log=False)
            bill_list.append(bill_data)
            actions = bill.export_actions()
            write_json(actions, join(OUTPUT_BASE_PATH, f'{bill.urlKey}--actions.json'), log=False)
            action_list.extend(actions)
            votes = bill.export_votes()
            write_json(votes, join(OUTPUT_BASE_PATH, f'{bill.urlKey}--votes.json'), log=False)
            vote_list.extend(votes)
        write_json(bill_list, join(OUTPUT_BASE_PATH, 'all-bills.json'))
        write_json(action_list, join(OUTPUT_BASE_PATH, 'all-bill-actions.json'))
        write_json(vote_list, join(OUTPUT_BASE_PATH, 'all-votes.json'))