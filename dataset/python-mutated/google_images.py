from __future__ import absolute_import, division, print_function, unicode_literals
__license__ = 'GPL v3'
__copyright__ = '2013, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
from collections import OrderedDict
from calibre import random_user_agent
from calibre.ebooks.metadata.sources.base import Source, Option

def parse_html(raw):
    if False:
        print('Hello World!')
    try:
        from html5_parser import parse
    except ImportError:
        import html5lib
        return html5lib.parse(raw, treebuilder='lxml', namespaceHTMLElements=False)
    else:
        return parse(raw)

def imgurl_from_id(raw, tbnid):
    if False:
        i = 10
        return i + 15
    from json import JSONDecoder
    q = '"{}",['.format(tbnid)
    start_pos = raw.index(q)
    if start_pos < 100:
        return
    jd = JSONDecoder()
    data = jd.raw_decode('[' + raw[start_pos:])[0]
    url_num = 0
    for x in data:
        if isinstance(x, list) and len(x) == 3:
            q = x[0]
            if hasattr(q, 'lower') and q.lower().startswith('http'):
                url_num += 1
                if url_num > 1:
                    return q

class GoogleImages(Source):
    name = 'Google Images'
    version = (1, 0, 5)
    minimum_calibre_version = (2, 80, 0)
    description = _('Downloads covers from a Google Image search. Useful to find larger/alternate covers.')
    capabilities = frozenset(['cover'])
    can_get_multiple_covers = True
    supports_gzip_transfer_encoding = True
    options = (Option('max_covers', 'number', 5, _('Maximum number of covers to get'), _('The maximum number of covers to process from the Google search result')), Option('size', 'choices', 'svga', _('Cover size'), _('Search for covers larger than the specified size'), choices=OrderedDict((('any', _('Any size')), ('l', _('Large')), ('qsvga', _('Larger than %s') % '400x300'), ('vga', _('Larger than %s') % '640x480'), ('svga', _('Larger than %s') % '600x800'), ('xga', _('Larger than %s') % '1024x768'), ('2mp', _('Larger than %s') % '2 MP'), ('4mp', _('Larger than %s') % '4 MP')))))

    def download_cover(self, log, result_queue, abort, title=None, authors=None, identifiers={}, timeout=30, get_best_cover=False):
        if False:
            while True:
                i = 10
        if not title:
            return
        timeout = max(60, timeout)
        title = ' '.join(self.get_title_tokens(title))
        author = ' '.join(self.get_author_tokens(authors))
        urls = self.get_image_urls(title, author, log, abort, timeout)
        self.download_multiple_covers(title, authors, urls, get_best_cover, timeout, result_queue, abort, log)

    @property
    def user_agent(self):
        if False:
            while True:
                i = 10
        return random_user_agent(allow_ie=False)

    def get_image_urls(self, title, author, log, abort, timeout):
        if False:
            return 10
        from calibre.utils.cleantext import clean_ascii_chars
        try:
            from urllib.parse import urlencode
        except ImportError:
            from urllib import urlencode
        from collections import OrderedDict
        ans = OrderedDict()
        br = self.browser
        q = urlencode({'as_q': ('%s %s' % (title, author)).encode('utf-8')})
        if isinstance(q, bytes):
            q = q.decode('utf-8')
        sz = self.prefs['size']
        if sz == 'any':
            sz = ''
        elif sz == 'l':
            sz = 'isz:l,'
        else:
            sz = 'isz:lt,islt:%s,' % sz
        url = 'https://www.google.com/search?as_st=y&tbm=isch&{}&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs={}iar:t,ift:jpg'.format(q, sz)
        log('Search URL: ' + url)
        br.set_simple_cookie('CONSENT', 'PENDING+987', '.google.com', path='/')
        template = b'\x08\x01\x128\x08\x14\x12+boq_identityfrontenduiserver_20231107.05_p0\x1a\x05en-US \x03\x1a\x06\x08\x80\xf1\xca\xaa\x06'
        from datetime import date
        from base64 import standard_b64encode
        template.replace(b'20231107', date.today().strftime('%Y%m%d').encode('ascii'))
        br.set_simple_cookie('SOCS', standard_b64encode(template).decode('ascii').rstrip('='), '.google.com', path='/')
        raw = clean_ascii_chars(br.open(url).read().decode('utf-8'))
        root = parse_html(raw)
        results = root.xpath('//div/@data-tbnid')
        for tbnid in results:
            try:
                imgurl = imgurl_from_id(raw, tbnid)
            except Exception:
                continue
            if imgurl:
                ans[imgurl] = True
        return list(ans)

def test():
    if False:
        i = 10
        return i + 15
    try:
        from queue import Queue
    except ImportError:
        from Queue import Queue
    from threading import Event
    from calibre.utils.logging import default_log
    p = GoogleImages(None)
    p.log = default_log
    rq = Queue()
    p.download_cover(default_log, rq, Event(), title='The Heroes', authors=('Joe Abercrombie',))
    print('Downloaded', rq.qsize(), 'covers')
if __name__ == '__main__':
    test()