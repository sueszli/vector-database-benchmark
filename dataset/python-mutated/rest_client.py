"""
Simple command line client into RPC commands
Can be used as an alternate to Telegram

Should not import anything from freqtrade,
so it can be used as a standalone script.
"""
import argparse
import inspect
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, urlunparse
import rapidjson
import requests
from requests.exceptions import ConnectionError
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ft_rest_client')

class FtRestClient:

    def __init__(self, serverurl, username=None, password=None):
        if False:
            for i in range(10):
                print('nop')
        self._serverurl = serverurl
        self._session = requests.Session()
        self._session.auth = (username, password)

    def _call(self, method, apipath, params: Optional[dict]=None, data=None, files=None):
        if False:
            i = 10
            return i + 15
        if str(method).upper() not in ('GET', 'POST', 'PUT', 'DELETE'):
            raise ValueError(f'invalid method <{method}>')
        basepath = f'{self._serverurl}/api/v1/{apipath}'
        hd = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        (schema, netloc, path, par, query, fragment) = urlparse(basepath)
        query = urlencode(params) if params else ''
        url = urlunparse((schema, netloc, path, par, query, fragment))
        try:
            resp = self._session.request(method, url, headers=hd, data=json.dumps(data))
            return resp.json()
        except ConnectionError:
            logger.warning('Connection error')

    def _get(self, apipath, params: Optional[dict]=None):
        if False:
            for i in range(10):
                print('nop')
        return self._call('GET', apipath, params=params)

    def _delete(self, apipath, params: Optional[dict]=None):
        if False:
            i = 10
            return i + 15
        return self._call('DELETE', apipath, params=params)

    def _post(self, apipath, params: Optional[dict]=None, data: Optional[dict]=None):
        if False:
            print('Hello World!')
        return self._call('POST', apipath, params=params, data=data)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        "Start the bot if it's in the stopped state.\n\n        :return: json object\n        "
        return self._post('start')

    def stop(self):
        if False:
            while True:
                i = 10
        'Stop the bot. Use `start` to restart.\n\n        :return: json object\n        '
        return self._post('stop')

    def stopbuy(self):
        if False:
            while True:
                i = 10
        'Stop buying (but handle sells gracefully). Use `reload_config` to reset.\n\n        :return: json object\n        '
        return self._post('stopbuy')

    def reload_config(self):
        if False:
            while True:
                i = 10
        'Reload configuration.\n\n        :return: json object\n        '
        return self._post('reload_config')

    def balance(self):
        if False:
            while True:
                i = 10
        'Get the account balance.\n\n        :return: json object\n        '
        return self._get('balance')

    def count(self):
        if False:
            i = 10
            return i + 15
        'Return the amount of open trades.\n\n        :return: json object\n        '
        return self._get('count')

    def entries(self, pair=None):
        if False:
            while True:
                i = 10
        'Returns List of dicts containing all Trades, based on buy tag performance\n        Can either be average for all pairs or a specific pair provided\n\n        :return: json object\n        '
        return self._get('entries', params={'pair': pair} if pair else None)

    def exits(self, pair=None):
        if False:
            i = 10
            return i + 15
        'Returns List of dicts containing all Trades, based on exit reason performance\n        Can either be average for all pairs or a specific pair provided\n\n        :return: json object\n        '
        return self._get('exits', params={'pair': pair} if pair else None)

    def mix_tags(self, pair=None):
        if False:
            return 10
        'Returns List of dicts containing all Trades, based on entry_tag + exit_reason performance\n        Can either be average for all pairs or a specific pair provided\n\n        :return: json object\n        '
        return self._get('mix_tags', params={'pair': pair} if pair else None)

    def locks(self):
        if False:
            for i in range(10):
                print('nop')
        'Return current locks\n\n        :return: json object\n        '
        return self._get('locks')

    def delete_lock(self, lock_id):
        if False:
            for i in range(10):
                print('nop')
        'Delete (disable) lock from the database.\n\n        :param lock_id: ID for the lock to delete\n        :return: json object\n        '
        return self._delete(f'locks/{lock_id}')

    def daily(self, days=None):
        if False:
            print('Hello World!')
        'Return the profits for each day, and amount of trades.\n\n        :return: json object\n        '
        return self._get('daily', params={'timescale': days} if days else None)

    def weekly(self, weeks=None):
        if False:
            while True:
                i = 10
        'Return the profits for each week, and amount of trades.\n\n        :return: json object\n        '
        return self._get('weekly', params={'timescale': weeks} if weeks else None)

    def monthly(self, months=None):
        if False:
            print('Hello World!')
        'Return the profits for each month, and amount of trades.\n\n        :return: json object\n        '
        return self._get('monthly', params={'timescale': months} if months else None)

    def edge(self):
        if False:
            while True:
                i = 10
        'Return information about edge.\n\n        :return: json object\n        '
        return self._get('edge')

    def profit(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the profit summary.\n\n        :return: json object\n        '
        return self._get('profit')

    def stats(self):
        if False:
            print('Hello World!')
        'Return the stats report (durations, sell-reasons).\n\n        :return: json object\n        '
        return self._get('stats')

    def performance(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the performance of the different coins.\n\n        :return: json object\n        '
        return self._get('performance')

    def status(self):
        if False:
            i = 10
            return i + 15
        'Get the status of open trades.\n\n        :return: json object\n        '
        return self._get('status')

    def version(self):
        if False:
            while True:
                i = 10
        'Return the version of the bot.\n\n        :return: json object containing the version\n        '
        return self._get('version')

    def show_config(self):
        if False:
            print('Hello World!')
        ' Returns part of the configuration, relevant for trading operations.\n        :return: json object containing the version\n        '
        return self._get('show_config')

    def ping(self):
        if False:
            print('Hello World!')
        'simple ping'
        configstatus = self.show_config()
        if not configstatus:
            return {'status': 'not_running'}
        elif configstatus['state'] == 'running':
            return {'status': 'pong'}
        else:
            return {'status': 'not_running'}

    def logs(self, limit=None):
        if False:
            i = 10
            return i + 15
        'Show latest logs.\n\n        :param limit: Limits log messages to the last <limit> logs. No limit to get the entire log.\n        :return: json object\n        '
        return self._get('logs', params={'limit': limit} if limit else 0)

    def trades(self, limit=None, offset=None):
        if False:
            print('Hello World!')
        'Return trades history, sorted by id\n\n        :param limit: Limits trades to the X last trades. Max 500 trades.\n        :param offset: Offset by this amount of trades.\n        :return: json object\n        '
        params = {}
        if limit:
            params['limit'] = limit
        if offset:
            params['offset'] = offset
        return self._get('trades', params)

    def trade(self, trade_id):
        if False:
            for i in range(10):
                print('nop')
        'Return specific trade\n\n        :param trade_id: Specify which trade to get.\n        :return: json object\n        '
        return self._get(f'trade/{trade_id}')

    def delete_trade(self, trade_id):
        if False:
            for i in range(10):
                print('nop')
        'Delete trade from the database.\n        Tries to close open orders. Requires manual handling of this asset on the exchange.\n\n        :param trade_id: Deletes the trade with this ID from the database.\n        :return: json object\n        '
        return self._delete(f'trades/{trade_id}')

    def cancel_open_order(self, trade_id):
        if False:
            for i in range(10):
                print('nop')
        'Cancel open order for trade.\n\n        :param trade_id: Cancels open orders for this trade.\n        :return: json object\n        '
        return self._delete(f'trades/{trade_id}/open-order')

    def whitelist(self):
        if False:
            i = 10
            return i + 15
        'Show the current whitelist.\n\n        :return: json object\n        '
        return self._get('whitelist')

    def blacklist(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Show the current blacklist.\n\n        :param add: List of coins to add (example: "BNB/BTC")\n        :return: json object\n        '
        if not args:
            return self._get('blacklist')
        else:
            return self._post('blacklist', data={'blacklist': args})

    def forcebuy(self, pair, price=None):
        if False:
            print('Hello World!')
        'Buy an asset.\n\n        :param pair: Pair to buy (ETH/BTC)\n        :param price: Optional - price to buy\n        :return: json object of the trade\n        '
        data = {'pair': pair, 'price': price}
        return self._post('forcebuy', data=data)

    def forceenter(self, pair, side, price=None):
        if False:
            print('Hello World!')
        "Force entering a trade\n\n        :param pair: Pair to buy (ETH/BTC)\n        :param side: 'long' or 'short'\n        :param price: Optional - price to buy\n        :return: json object of the trade\n        "
        data = {'pair': pair, 'side': side}
        if price:
            data['price'] = price
        return self._post('forceenter', data=data)

    def forceexit(self, tradeid, ordertype=None, amount=None):
        if False:
            while True:
                i = 10
        'Force-exit a trade.\n\n        :param tradeid: Id of the trade (can be received via status command)\n        :param ordertype: Order type to use (must be market or limit)\n        :param amount: Amount to sell. Full sell if not given\n        :return: json object\n        '
        return self._post('forceexit', data={'tradeid': tradeid, 'ordertype': ordertype, 'amount': amount})

    def strategies(self):
        if False:
            while True:
                i = 10
        'Lists available strategies\n\n        :return: json object\n        '
        return self._get('strategies')

    def strategy(self, strategy):
        if False:
            print('Hello World!')
        'Get strategy details\n\n        :param strategy: Strategy class name\n        :return: json object\n        '
        return self._get(f'strategy/{strategy}')

    def pairlists_available(self):
        if False:
            while True:
                i = 10
        'Lists available pairlist providers\n\n        :return: json object\n        '
        return self._get('pairlists/available')

    def plot_config(self):
        if False:
            return 10
        'Return plot configuration if the strategy defines one.\n\n        :return: json object\n        '
        return self._get('plot_config')

    def available_pairs(self, timeframe=None, stake_currency=None):
        if False:
            return 10
        'Return available pair (backtest data) based on timeframe / stake_currency selection\n\n        :param timeframe: Only pairs with this timeframe available.\n        :param stake_currency: Only pairs that include this timeframe\n        :return: json object\n        '
        return self._get('available_pairs', params={'stake_currency': stake_currency if timeframe else '', 'timeframe': timeframe if timeframe else ''})

    def pair_candles(self, pair, timeframe, limit=None):
        if False:
            for i in range(10):
                print('nop')
        'Return live dataframe for <pair><timeframe>.\n\n        :param pair: Pair to get data for\n        :param timeframe: Only pairs with this timeframe available.\n        :param limit: Limit result to the last n candles.\n        :return: json object\n        '
        params = {'pair': pair, 'timeframe': timeframe}
        if limit:
            params['limit'] = limit
        return self._get('pair_candles', params=params)

    def pair_history(self, pair, timeframe, strategy, timerange=None, freqaimodel=None):
        if False:
            while True:
                i = 10
        'Return historic, analyzed dataframe\n\n        :param pair: Pair to get data for\n        :param timeframe: Only pairs with this timeframe available.\n        :param strategy: Strategy to analyze and get values for\n        :param freqaimodel: FreqAI model to use for analysis\n        :param timerange: Timerange to get data for (same format than --timerange endpoints)\n        :return: json object\n        '
        return self._get('pair_history', params={'pair': pair, 'timeframe': timeframe, 'strategy': strategy, 'freqaimodel': freqaimodel, 'timerange': timerange if timerange else ''})

    def sysinfo(self):
        if False:
            while True:
                i = 10
        'Provides system information (CPU, RAM usage)\n\n        :return: json object\n        '
        return self._get('sysinfo')

    def health(self):
        if False:
            while True:
                i = 10
        'Provides a quick health check of the running bot.\n\n        :return: json object\n        '
        return self._get('health')

def add_arguments():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='Positional argument defining the command to execute.', nargs='?')
    parser.add_argument('--show', help='Show possible methods with this client', dest='show', action='store_true', default=False)
    parser.add_argument('-c', '--config', help='Specify configuration file (default: %(default)s). ', dest='config', type=str, metavar='PATH', default='config.json')
    parser.add_argument('command_arguments', help='Positional arguments for the parameters for [command]', nargs='*', default=[])
    args = parser.parse_args()
    return vars(args)

def load_config(configfile):
    if False:
        print('Hello World!')
    file = Path(configfile)
    if file.is_file():
        with file.open('r') as f:
            config = rapidjson.load(f, parse_mode=rapidjson.PM_COMMENTS | rapidjson.PM_TRAILING_COMMAS)
        return config
    else:
        logger.warning(f'Could not load config file {file}.')
        sys.exit(1)

def print_commands():
    if False:
        i = 10
        return i + 15
    client = FtRestClient(None)
    print('Possible commands:\n')
    for (x, y) in inspect.getmembers(client):
        if not x.startswith('_'):
            doc = re.sub(':return:.*', '', getattr(client, x).__doc__, flags=re.MULTILINE).rstrip()
            print(f'{x}\n\t{doc}\n')

def main(args):
    if False:
        while True:
            i = 10
    if args.get('show'):
        print_commands()
        sys.exit()
    config = load_config(args['config'])
    url = config.get('api_server', {}).get('listen_ip_address', '127.0.0.1')
    port = config.get('api_server', {}).get('listen_port', '8080')
    username = config.get('api_server', {}).get('username')
    password = config.get('api_server', {}).get('password')
    server_url = f'http://{url}:{port}'
    client = FtRestClient(server_url, username, password)
    m = [x for (x, y) in inspect.getmembers(client) if not x.startswith('_')]
    command = args['command']
    if command not in m:
        logger.error(f'Command {command} not defined')
        print_commands()
        return
    print(json.dumps(getattr(client, command)(*args['command_arguments'])))
if __name__ == '__main__':
    args = add_arguments()
    main(args)