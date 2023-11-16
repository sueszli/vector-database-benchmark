import json
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from PyPDF2 import PdfReader
from os.path import exists, join
from config import SESSION_ID, CACHE_BASE_PATH
FLOOR_DATE_FORMAT = '%B %m, %Y'

class Vote:
    """Data structure for Montana Legislature vote

        Can be a committtee, floor or mail veto override vote.
    """

    def __init__(self, inputs, bill_needs_refresh=False, use_verbose_logging=False, cache_base_path=CACHE_BASE_PATH):
        if False:
            i = 10
            return i + 15
        self.id = inputs['action_id']
        self.inputs = inputs
        self.cache_base_path = cache_base_path
        self.bill_needs_refresh = bill_needs_refresh
        self.use_cache = True
        self.use_verbose_logging = use_verbose_logging
        self.data = {'url': inputs['url'], 'bill': inputs['bill'], 'session': SESSION_ID, 'action_id': inputs['action_id'], 'type': inputs['type']}
        if inputs['type'] == 'veto override':
            self.parse_override_vote()
        if inputs['type'] == 'floor':
            self.parse_floor_vote(inputs['url'])
        if inputs['type'] == 'committee':
            self.parse_committee_vote(inputs['url'])

    def parse_floor_vote(self, url):
        if False:
            i = 10
            return i + 15
        '\n        Parse HTML floor vote page,\n        e.g. http://laws.leg.mt.gov/legprd/LAW0211W$BLAC.VoteTabulation?P_VOTE_SEQ=H2050&P_SESS=20211\n        '
        CACHE_PATH = join(self.cache_base_path, 'votes', f'{self.id}.html')
        if exists(CACHE_PATH) and self.use_cache:
            if self.use_verbose_logging:
                print('--- Reading floor vote data from', CACHE_PATH)
            with open(CACHE_PATH, 'r') as f:
                text = f.read()
        elif not exists(CACHE_PATH) and (not self.bill_needs_refresh):
            print(f'ooo Skipping missing floor data for {self.id} from', url)
            self.data['totals'] = self.inputs['bill_page_vote_count']
            self.data['error'] = 'Skipped previously missing vote page'
            return None
        elif not url:
            self.data['totals'] = self.inputs['bill_page_vote_count']
            self.data['error'] = 'Missing vote page'
            return None
        else:
            if self.use_verbose_logging:
                print(f'+++ Fetching floor vote data for {self.id} from', url)
            response = requests.get(url)
            text = response.text
            if 'No Vote Records Found for this Action.' in text:
                self.data['totals'] = self.inputs['bill_page_vote_count']
                self.data['error'] = 'Missing vote page'
                err = self.data['error']
                if self.use_verbose_logging:
                    print(f'  * {err} fetching {self.id}. URL:', url)
                return None
            if self.use_verbose_logging:
                print('--- Writing floor vote data to cache', CACHE_PATH)
            with open(CACHE_PATH, 'w') as f:
                f.write(text)
        soup = BeautifulSoup(text, 'lxml')
        if not url:
            if 'MONTANA SENATE' in text:
                self.data['seq_number'] = 'SXXX'
            elif 'MONTANA HOUSE' in text:
                self.data['seq_number'] = 'HXXX'
            else:
                self.data['seq_number'] = 'error'
        else:
            self.data['seq_number'] = re.search('(?<=VOTE_SEQ\\=)(H|S)\\d+', url).group(0)
        vote_date = soup.find(text=re.compile('DATE:')).text.replace('DATE:', '').strip()
        self.data['date'] = vote_date
        vote_description = soup.find_all('p')[1].text.strip()
        self.data['description'] = vote_description
        total_table = soup.find(text='YEAS').find_parent('table')
        total_cells = total_table.find_all('tr')[1].find_all('td')
        self.data['totals'] = {'Y': int(total_cells[0].text), 'N': int(total_cells[1].text), 'E': int(total_cells[2].text), 'A': int(total_cells[3].text)}
        vote_cells = soup.find_all('table')[2].find_all('td')
        votes_by_name = []
        for td in vote_cells:
            text = td.text
            if len(text.strip()) == 0:
                continue
            votes_by_name.append({'name': re.search('(?<=^(Y|N|E|A)).+', text).group(0).strip(), 'vote': re.search('^(Y|N|E|A)', text).group(0)})
        self.data['votes'] = votes_by_name

    def parse_committee_vote(self, url):
        if False:
            while True:
                i = 10
        '\n        Parse PDF committee vote page,\n        e.g. https://leg.mt.gov/bills/2021/minutes/house/votesheets/HB0701TAH210401.pdf\n        '
        CACHE_PATH = join(self.cache_base_path, 'votes', f'{self.id}.pdf')
        self.data['seq_number'] = None
        self.data['error'] = None
        if exists(CACHE_PATH) and self.use_cache:
            if self.use_verbose_logging:
                print('--- Reading committee vote data from', CACHE_PATH)
        elif not exists(CACHE_PATH) and (not self.bill_needs_refresh):
            print(f'ooo Skipping missing committee vote data for {self.id}')
            self.data['totals'] = self.inputs['bill_page_vote_count']
            self.data['error'] = 'Skipped previously missing vote page'
            return None
        elif url is not None:
            if self.use_verbose_logging:
                print('+++ Fetching committee vote data from URL', url)
            response = requests.get(url)
            if response.status_code == 200:
                raw = response.content
                with open(CACHE_PATH, 'wb') as f:
                    f.write(raw)
            else:
                self.data['totals'] = self.inputs['bill_page_vote_count']
                self.data['error'] = 'Missing PDF'
        else:
            self.data['totals'] = self.inputs['bill_page_vote_count']
            self.data['error'] = 'Missing URL'
        if self.data['error'] is not None:
            err = self.data['error']
            if self.use_verbose_logging:
                print(f'  * {err} fetching {self.id}. URL:', url)
            return None
        with open(CACHE_PATH, 'rb') as f:
            pdf = PdfReader(f)
            text = pdf.getPage(0).extractText()
            header_rows = re.search('(?s).+(?=\\nYEAS\\s+(\\-|–)\\s+[0-9]+\\s+NAYS\\s+(\\-|–)\\s+[0-9]+)', text).group(0).split('\n')
            total_row = re.search('YEAS\\s+(\\-|–)\\s+[0-9]+\\s+NAYS\\s+(\\-|–)\\s+[0-9]+', text).group(0)
            vote_re = re.compile('(Y|N|E|A).+')
            vote_rows = list(filter(vote_re.match, text.split('\n')[len(header_rows) + 1:]))
            self.data['date'] = header_rows[1]
            self.data['description'] = header_rows[-1]
            self.data['totals'] = {'Y': int(re.search('(?<=YEAS (\\-|–) )\\d+', total_row.replace('  ', ' ')).group(0)), 'N': int(re.search('(?<=NAYS (\\-|–) )\\d+', total_row.replace('  ', ' ')).group(0))}
            votes_by_name = []
            for row in vote_rows:
                votes_by_name.append({'name': re.search('(?<=^(Y|N|E|A)).+', row).group(0).strip().replace(' ', '').replace(',', ', ').replace(';byProxy', ''), 'vote': re.search('^(Y|N|E|A)', row).group(0)})
            self.data['votes'] = votes_by_name

    def parse_override_vote(self):
        if False:
            while True:
                i = 10
        self.data['seq_number'] = None
        self.data['error'] = None
        self.data['date'] = self.inputs['action_date']
        self.data['description'] = self.inputs['action_description']
        self.data['totals'] = self.inputs['bill_page_vote_count']

    def check_vote_counts_match(self, totals, names, bill_page_totals):
        if False:
            while True:
                i = 10
        pass

    def export(self):
        if False:
            i = 10
            return i + 15
        return self.data