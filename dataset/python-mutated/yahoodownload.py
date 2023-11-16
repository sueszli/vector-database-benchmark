from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import collections
import datetime
import io
import logging
import sys
PY2 = sys.version_info.major == 2
if PY2:
    from urllib2 import urlopen
    from urllib import quote as urlquote
else:
    from urllib.request import urlopen
    from urllib.parse import quote as urlquote
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class YahooDownload(object):
    urlhist = 'https://finance.yahoo.com/quote/{}/history'
    urldown = 'https://query1.finance.yahoo.com/v7/finance/download'
    retries = 3

    def __init__(self, ticker, fromdate, todate, period='d', reverse=False):
        if False:
            for i in range(10):
                print('nop')
        try:
            import requests
        except ImportError:
            msg = 'The new Yahoo data feed requires to have the requests module installed. Please use pip install requests or the method of your choice'
            raise Exception(msg)
        url = self.urlhist.format(ticker)
        sesskwargs = dict()
        if False and self.p.proxies:
            sesskwargs['proxies'] = self.p.proxies
        crumb = None
        sess = requests.Session()
        for i in range(self.retries + 1):
            resp = sess.get(url, **sesskwargs)
            if resp.status_code != requests.codes.ok:
                continue
            txt = resp.text
            i = txt.find('CrumbStore')
            if i == -1:
                continue
            i = txt.find('crumb', i)
            if i == -1:
                continue
            istart = txt.find('"', i + len('crumb') + 1)
            if istart == -1:
                continue
            istart += 1
            iend = txt.find('"', istart)
            if iend == -1:
                continue
            crumb = txt[istart:iend]
            crumb = crumb.encode('ascii').decode('unicode-escape')
            break
        if crumb is None:
            self.error = 'Crumb not found'
            self.f = None
            return
        urld = '{}/{}'.format(self.urldown, ticker)
        urlargs = []
        posix = datetime.date(1970, 1, 1)
        if todate is not None:
            period2 = (todate.date() - posix).total_seconds()
            urlargs.append('period2={}'.format(int(period2)))
        if todate is not None:
            period1 = (fromdate.date() - posix).total_seconds()
            urlargs.append('period1={}'.format(int(period1)))
        intervals = {'d': '1d', 'w': '1wk', 'm': '1mo'}
        urlargs.append('interval={}'.format(intervals[period]))
        urlargs.append('events=history')
        urlargs.append('crumb={}'.format(crumb))
        urld = '{}?{}'.format(urld, '&'.join(urlargs))
        f = None
        for i in range(self.retries + 1):
            resp = sess.get(urld, **sesskwargs)
            if resp.status_code != requests.codes.ok:
                continue
            ctype = resp.headers['Content-Type']
            if 'text/csv' not in ctype:
                self.error = 'Wrong content type: %s' % ctype
                continue
            try:
                f = io.StringIO(resp.text, newline=None)
            except Exception:
                continue
            break
        self.datafile = f

    def writetofile(self, filename):
        if False:
            i = 10
            return i + 15
        if not self.datafile:
            return
        if not hasattr(filename, 'read'):
            f = io.open(filename, 'w')
        else:
            f = filename
        self.datafile.seek(0)
        for line in self.datafile:
            f.write(line)
        f.close()

def parse_args():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Download Yahoo CSV Finance Data')
    parser.add_argument('--ticker', required=True, help='Ticker to be downloaded')
    parser.add_argument('--reverse', action='store_true', default=False, help='Do reverse the downloaded files')
    parser.add_argument('--timeframe', default='d', help='Timeframe: d -> day, w -> week, m -> month')
    parser.add_argument('--fromdate', required=True, help='Starting date in YYYY-MM-DD format')
    parser.add_argument('--todate', required=True, help='Ending date in YYYY-MM-DD format')
    parser.add_argument('--outfile', required=True, help='Output file name')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    logging.info('Processing input parameters')
    logging.info('Processing fromdate')
    try:
        fromdate = datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')
    except Exception as e:
        logging.error('Converting fromdate failed')
        logging.error(str(e))
        sys.exit(1)
    logging.info('Processing todate')
    try:
        todate = datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    except Exception as e:
        logging.error('Converting todate failed')
        logging.error(str(e))
        sys.exit(1)
    logging.info('Do Not Reverse flag status')
    reverse = args.reverse
    logging.info('Downloading from yahoo')
    try:
        yahoodown = YahooDownload(ticker=args.ticker, fromdate=fromdate, todate=todate, period=args.timeframe, reverse=reverse)
    except Exception as e:
        logging.error('Downloading data from Yahoo failed')
        logging.error(str(e))
        sys.exit(1)
    logging.info('Opening output file')
    try:
        ofile = io.open(args.outfile, 'w')
    except IOError as e:
        logging.error('Error opening output file')
        logging.error(str(e))
        sys.exit(1)
    logging.info('Writing downloaded data to output file')
    try:
        yahoodown.writetofile(ofile)
    except Exception as e:
        logging.error('Writing to output file failed')
        logging.error(str(e))
        sys.exit(1)
    logging.info('All operations completed successfully')
    sys.exit(0)