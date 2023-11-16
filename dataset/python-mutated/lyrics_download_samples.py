import os
import sys
from test.plugins import test_lyrics
import requests

def mkdir_p(path):
    if False:
        i = 10
        return i + 15
    try:
        os.makedirs(path)
    except OSError:
        if os.path.isdir(path):
            pass
        else:
            raise

def safe_open_w(path):
    if False:
        return 10
    'Open "path" for writing, creating any parent directories as needed.'
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')

def main(argv=None):
    if False:
        i = 10
        return i + 15
    'Download one lyrics sample page per referenced source.'
    if argv is None:
        argv = sys.argv
    print('Fetching samples from:')
    for s in test_lyrics.GOOGLE_SOURCES + test_lyrics.DEFAULT_SOURCES:
        print(s['url'])
        url = s['url'] + s['path']
        fn = test_lyrics.url_to_filename(url)
        if not os.path.isfile(fn):
            html = requests.get(url, verify=False).text
            with safe_open_w(fn) as f:
                f.write(html.encode('utf-8'))
if __name__ == '__main__':
    sys.exit(main())