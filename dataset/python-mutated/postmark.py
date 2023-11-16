"""PayPal API Client."""
import logging
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Callable, Generator, List
import httpx
import singer
from dateutil.rrule import DAILY, rrule
from mage_integrations.sources.postmark.tap_postmark.cleaners import CLEANERS
API_SCHEME: str = 'https://'
API_BASE_URL: str = 'api.postmarkapp.com'
API_MESSAGES_PATH: str = '/messages'
API_STATS_PATH: str = '/stats'
API_OUTBOUND_PATH: str = '/outbound'
API_OPENS_PATH: str = '/opens'
API_BOUNCE_PATH: str = '/bounces'
API_CLIENTS_PATH: str = '/emailclients'
API_PLATFORM_PATH: str = '/platforms'
API_DATE_PATH: str = '?fromdate=:date:&todate=:date:'
MESSAGES_MAX_HISTORY: timedelta = timedelta(days=45)
HEADERS: MappingProxyType = MappingProxyType({'Accept': 'application/json', 'X-Postmark-Server-Token': ':token:'})

class Postmark(object):
    """Postmark API Client."""

    def __init__(self, postmark_server_token: str) -> None:
        if False:
            while True:
                i = 10
        'Initialize client.\n\n        Arguments:\n            postmark_server_token {str} -- Postmark Server Token\n        '
        self.postmark_server_token: str = postmark_server_token
        self.logger: logging.Logger = singer.get_logger()
        self.client: httpx.Client = httpx.Client(http2=True)

    def stats_outbound_bounces(self, **kwargs: dict) -> Generator[dict, None, None]:
        if False:
            while True:
                i = 10
        'Get all bounce reasons from date.\n\n        Raises:\n            ValueError: When the parameter start_date is missing\n\n        Yields:\n            Generator[dict] --  Cleaned Bounce Data\n        '
        start_date_input: str = str(kwargs.get('start_date', ''))
        if not start_date_input:
            raise ValueError('The parameter start_date is required.')
        cleaner: Callable = CLEANERS.get('postmark_stats_outbound_bounces', {})
        self._create_headers()
        for date_day in self._start_days_till_now(start_date_input):
            from_to_date: str = API_DATE_PATH.replace(':date:', date_day)
            self.logger.info(f'Recieving Bounce stats from {date_day}')
            url: str = f'{API_SCHEME}{API_BASE_URL}{API_STATS_PATH}{API_OUTBOUND_PATH}{API_BOUNCE_PATH}{from_to_date}'
            response: httpx._models.Response = self.client.get(url, headers=self.headers)
            response.raise_for_status()
            response_data: dict = response.json()
            yield cleaner(date_day, response_data)

    def stats_outbound_clients(self, **kwargs: dict) -> Generator[dict, None, None]:
        if False:
            return 10
        'Get all clients from date.\n\n        Raises:\n            ValueError: When the parameter start_date is missing\n\n        Yields:\n            Generator[dict] --  Cleaned Client Data\n        '
        start_date_input: str = str(kwargs.get('start_date', ''))
        if not start_date_input:
            raise ValueError('The parameter start_date is required.')
        cleaner: Callable = CLEANERS.get('postmark_stats_outbound_clients', {})
        self._create_headers()
        for date_day in self._start_days_till_now(start_date_input):
            from_to_date: str = API_DATE_PATH.replace(':date:', date_day)
            self.logger.info(f'Recieving Client stats from {date_day}')
            url: str = f'{API_SCHEME}{API_BASE_URL}{API_STATS_PATH}{API_OUTBOUND_PATH}{API_OPENS_PATH}{API_CLIENTS_PATH}{from_to_date}'
            response: httpx._models.Response = self.client.get(url, headers=self.headers)
            response.raise_for_status()
            response_data: dict = response.json()
            yield from cleaner(date_day, response_data)

    def stats_outbound_overview(self, **kwargs: dict) -> Generator[dict, None, None]:
        if False:
            while True:
                i = 10
        'Get all bounce reasons from date.\n\n        Raises:\n            ValueError: When the parameter start_date is missing\n\n        Yields:\n            Generator[dict] --  Cleaned Bounce Data\n        '
        start_date_input: str = str(kwargs.get('start_date', ''))
        if not start_date_input:
            raise ValueError('The parameter start_date is required.')
        cleaner: Callable = CLEANERS.get('postmark_stats_outbound_overview', {})
        self._create_headers()
        for date_day in self._start_days_till_now(start_date_input):
            from_to_date: str = API_DATE_PATH.replace(':date:', date_day)
            self.logger.info(f'Recieving overview stats from {date_day}')
            url: str = f'{API_SCHEME}{API_BASE_URL}{API_STATS_PATH}{API_OUTBOUND_PATH}{from_to_date}'
            response: httpx._models.Response = self.client.get(url, headers=self.headers)
            response.raise_for_status()
            response_data: dict = response.json()
            yield cleaner(date_day, response_data)

    def stats_outbound_platform(self, **kwargs: dict) -> Generator[dict, None, None]:
        if False:
            i = 10
            return i + 15
        'Get all platforms that opened mails from date.\n\n        Raises:\n            ValueError: When the parameter start_date is missing\n\n        Yields:\n            Generator[dict] --  Cleaned Bounce Data\n        '
        start_date_input: str = str(kwargs.get('start_date', ''))
        if not start_date_input:
            raise ValueError('The parameter start_date is required.')
        cleaner: Callable = CLEANERS.get('postmark_stats_outbound_platform', {})
        self._create_headers()
        for date_day in self._start_days_till_now(start_date_input):
            from_to_date: str = API_DATE_PATH.replace(':date:', date_day)
            self.logger.info(f'Recieving platform opens from {date_day}')
            url: str = f'{API_SCHEME}{API_BASE_URL}{API_STATS_PATH}{API_OUTBOUND_PATH}{API_OPENS_PATH}{API_PLATFORM_PATH}{from_to_date}'
            response: httpx._models.Response = self.client.get(url, headers=self.headers)
            response.raise_for_status()
            response_data: dict = response.json()
            yield cleaner(date_day, response_data)

    def messages_outbound(self, **kwargs: dict) -> Generator[dict, None, None]:
        if False:
            return 10
        'Outbound messages.\n\n        Raises:\n            ValueError: When the parameter start_date is not in the kwargs\n            ValueError: If the start_date is more than 45 days ago\n\n        Yields:\n            Generator[dict, None, None] -- Messages\n        '
        start_date_input: str = str(kwargs.get('start_date', ''))
        if not start_date_input:
            raise ValueError('The parameter start_date is required.')
        start_date: date = datetime.strptime(start_date_input, '%Y-%m-%d').date()
        if start_date < date.today() - MESSAGES_MAX_HISTORY:
            raise ValueError('The start_date must be at max 45 days ago.')
        cleaner: Callable = CLEANERS.get('postmark_messages_outbound', {})
        self._create_headers()
        url: str = f'{API_SCHEME}{API_BASE_URL}{API_MESSAGES_PATH}{API_OUTBOUND_PATH}'
        batch_size: int = 500
        for date_day in self._start_days_till_now(start_date_input):
            http_parameters: dict = {'count': batch_size, 'fromdate': date_day, 'todate': date_day, 'offset': 0}
            more = True
            total = 0
            while more:
                counter: int = total // batch_size + 1
                response: httpx._models.Response = self.client.get(url, headers=self.headers, params=http_parameters)
                response.raise_for_status()
                response_data: dict = response.json()
                message_data: List[dict] = response_data.get('Messages', [])
                message_count: int = len(message_data)
                if message_count < batch_size:
                    more = False
                for message in message_data:
                    API_MESSAGEID: str = '/' + message['MessageID']
                    API_DETAILS = '/details'
                    url2: str = f'{API_SCHEME}{API_BASE_URL}{API_MESSAGES_PATH}{API_OUTBOUND_PATH}{API_MESSAGEID}{API_DETAILS}'
                    details_response: httpx._models.Response = self.client.get(url2, headers=self.headers, params=http_parameters)
                    details_response.raise_for_status()
                    details_data: dict = details_response.json()
                    details_messageEvents: List[dict] = details_data.get('MessageEvents', '')
                    messageEvent_types: str = ''
                    for event in details_messageEvents:
                        messageEvent_types = messageEvent_types + event['Type'] + ','
                    messageEvent_types = messageEvent_types[:-1]
                    message['MessageEvents'] = messageEvent_types
                    yield cleaner(date_day, message)
                    total += 1
                self.logger.info(f'Date {date_day}, batch: {counter}, messages: {total}')
                http_parameters['offset'] += batch_size

    def messages_opens(self, **kwargs: dict) -> Generator[dict, None, None]:
        if False:
            i = 10
            return i + 15
        'Opens messages.\n\n        Raises:\n            ValueError: When the parameter start_date is not in the kwargs\n            ValueError: If the start_date is more than 45 days ago\n\n        Yields:\n            Generator[dict, None, None] -- Messages\n        '
        start_date_input: str = str(kwargs.get('start_date', ''))
        if not start_date_input:
            raise ValueError('The parameter start_date is required.')
        start_date: date = datetime.strptime(start_date_input, '%Y-%m-%d').date()
        if start_date < date.today() - MESSAGES_MAX_HISTORY:
            raise ValueError('The start_date must be at max 45 days ago.')
        cleaner: Callable = CLEANERS.get('postmark_messages_opens', {})
        self._create_headers()
        url: str = f'{API_SCHEME}{API_BASE_URL}{API_MESSAGES_PATH}{API_OUTBOUND_PATH}{API_OPENS_PATH}'
        batch_size: int = 500
        for date_day in self._start_days_till_now(start_date_input):
            http_parameters: dict = {'count': batch_size, 'fromdate': date_day, 'todate': date_day, 'offset': 0}
            more = True
            total = 0
            while more:
                counter: int = total // batch_size + 1
                response: httpx._models.Response = self.client.get(url, headers=self.headers, params=http_parameters)
                response.raise_for_status()
                response_data: dict = response.json()
                message_data: List[dict] = response_data.get('Opens', [])
                message_count: int = len(message_data)
                if message_count < batch_size:
                    more = False
                for message in message_data:
                    yield cleaner(date_day, message)
                    total += 1
                self.logger.info(f'Date {date_day}, batch: {counter}, opens: {total}')
                http_parameters['offset'] += batch_size
                if http_parameters['offset'] >= 10000:
                    break

    def _start_days_till_now(self, start_date: str) -> Generator:
        if False:
            print('Hello World!')
        'Yield YYYY/MM/DD for every day until now.\n\n        Arguments:\n            start_date {str} -- Start date e.g. 2020-01-01\n\n        Yields:\n            Generator -- Every day until now.\n        '
        year: int = int(start_date.split('-')[0])
        month: int = int(start_date.split('-')[1].lstrip())
        day: int = int(start_date.split('-')[2].lstrip())
        period: date = date(year, month, day)
        dates: rrule = rrule(freq=DAILY, dtstart=period, until=datetime.utcnow())
        yield from (date_day.strftime('%Y-%m-%d') for date_day in dates)

    def _create_headers(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create authentication headers for requests.'
        headers: dict = dict(HEADERS)
        headers['X-Postmark-Server-Token'] = headers['X-Postmark-Server-Token'].replace(':token:', self.postmark_server_token)
        self.headers = headers