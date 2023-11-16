import pytest
from boltons import urlutils
from boltons.urlutils import URL, _URL_RE, find_all_links
try:
    unicode
except NameError:
    unicode = str
TEST_URLS = ['http://googlewebsite.com/e-shops.aspx', 'http://example.com:8080/search?q=123&business=Nothing%20Special', 'http://hatnote.com:9000?arg=1&arg=2&arg=3', 'https://xn--bcher-kva.ch', 'http://xn--ggbla1c4e.xn--ngbc5azd/', 'http://tools.ietf.org/html/rfc3986#section-3.4', 'http://wiki:pedia@hatnote.com', 'ftp://ftp.rfc-editor.org/in-notes/tar/RFCs0001-0500.tar.gz', 'http://[1080:0:0:0:8:800:200C:417A]/index.html', 'ssh://192.0.2.16:2222/', 'https://[::101.45.75.219]:80/?hi=bye', 'ldap://[::192.9.5.5]/dc=example,dc=com??sub?(sn=Jensen)', 'mailto:me@example.com?to=me@example.com&body=hi%20http://wikipedia.org', 'news:alt.rec.motorcycle', 'tel:+1-800-867-5309', 'urn:oasis:member:A00024:x', 'magnet:?xt=urn:btih:1a42b9e04e122b97a5254e3df77ab3c4b7da725f&dn=Puppy%20Linux%20precise-5.7.1.iso&tr=udp://tracker.openbittorrent.com:80&tr=udp://tracker.publicbt.com:80&tr=udp://tracker.istole.it:6969&tr=udp://tracker.ccc.de:80&tr=udp://open.demonii.com:1337', 'http://localhost', 'http://localhost/', 'http://localhost/foo', 'http://localhost/foo/', 'http://localhost/foo!!bar/', 'http://localhost/foo%20bar/', 'http://localhost/foo%2Fbar/', 'http://localhost/foo?n', 'http://localhost/foo?n=v', 'http://localhost/foo?n=/a/b', 'http://example.com/foo!@$bar?b!@z=123', 'http://localhost/asd?a=asd%20sdf/345', 'http://(%2525)/(%2525)?(%2525)&(%2525)=(%2525)#(%2525)', 'http://(%C3%A9)/(%C3%A9)?(%C3%A9)&(%C3%A9)=(%C3%A9)#(%C3%A9)']

@pytest.fixture(scope='module', params=TEST_URLS)
def test_url(request):
    if False:
        i = 10
        return i + 15
    param = request.param
    return param

def test_regex(test_url):
    if False:
        while True:
            i = 10
    match = _URL_RE.match(test_url)
    assert match.groupdict()

def test_roundtrip(test_url):
    if False:
        print('Hello World!')
    result = URL(test_url).to_text(full_quote=True)
    assert test_url == result

def test_basic():
    if False:
        i = 10
        return i + 15
    u1 = URL('http://googlewebsite.com/e-shops.aspx')
    assert isinstance(u1.to_text(), unicode)
    assert u1.host == 'googlewebsite.com'

def test_utf8_url():
    if False:
        print('Hello World!')
    url_bytes = b'http://\xd9\x85\xd8\xab\xd8\xa7\xd9\x84.\xd8\xa2\xd8\xb2\xd9\x85\xd8\xa7\xdb\x8c\xd8\xb4\xdb\x8c'
    url = URL(url_bytes)
    assert url.scheme == 'http'
    assert url.host == u'مثال.آزمایشی'

def test_idna():
    if False:
        print('Hello World!')
    u1 = URL(u'http://bücher.ch')
    assert u1.host == u'bücher.ch'
    assert u1.to_text(full_quote=True) == 'http://xn--bcher-kva.ch'
    assert u1.to_text(full_quote=False) == u'http://bücher.ch'
    u2 = URL('https://xn--bcher-kva.ch')
    assert u2.host == u'bücher.ch'
    assert u2.to_text(full_quote=True) == 'https://xn--bcher-kva.ch'
    assert u2.to_text(full_quote=False) == u'https://bücher.ch'

def test_query_params(test_url):
    if False:
        return 10
    url_obj = URL(test_url)
    if not url_obj.query_params or url_obj.fragment:
        return
    qp_text = url_obj.query_params.to_text(full_quote=True)
    assert test_url.endswith(qp_text)

def test_iri_query():
    if False:
        i = 10
        return i + 15
    url = URL(u'http://minerals.mountain.ore/?rock=☘')
    assert url.query_params['rock'] == u'☘'
    assert url.query_params.to_text(full_quote=True).endswith(u'%E2%98%98')

def test_iri_path():
    if False:
        return 10
    url = URL(u'http://minerals.mountain.ore/rock/☘/')
    assert url.path == u'/rock/☘/'
    assert url.to_text(full_quote=True).endswith('%E2%98%98/')

def test_url_copy():
    if False:
        while True:
            i = 10
    url = URL('http://example.com/foo?bar=baz')
    url_copy = URL(url)
    assert url == url_copy

def test_invalid_port():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        URL('http://reader.googlewebsite.com:neverforget')

def test_invalid_ipv6():
    if False:
        i = 10
        return i + 15
    invalid_ipv6_ips = ['2001::0234:C1ab::A0:aabc:003F', '2001::1::3F']
    for ip in invalid_ipv6_ips:
        with pytest.raises(ValueError):
            URL('http://[' + ip + ']')

def test_parse_url():
    if False:
        while True:
            i = 10
    expected = {'family': 2, 'password': None, 'fragment': None, 'authority': u'127.0.0.1:3000', 'port': 3000, 'query': u'a=1', '_netloc_sep': u'//', 'path': u'/', 'scheme': u'http', 'host': u'127.0.0.1', 'username': None}
    res = urlutils.parse_url('http://127.0.0.1:3000/?a=1')
    assert res == expected

def test_parse_equals_in_qp_value():
    if False:
        return 10
    u = URL('http://localhost/?=x=x=x')
    assert u.qp[''] == 'x=x=x'
    assert u.to_text() == 'http://localhost/?=x%3Dx%3Dx'
    u = URL('http://localhost/?foo=x=x=x&bar=y')
    assert u.qp['foo'] == 'x=x=x'
    assert u.qp['bar'] == 'y'

def test_identical_equal():
    if False:
        print('Hello World!')
    u = URL('http://example.com/path?query=param#frag')
    assert u == u

def test_equal():
    if False:
        for i in range(10):
            print('nop')
    u = URL('http://example.com/path?query=param#frag')
    bono = URL('http://example.com/path?query=param#frag')
    assert bono == u

def test_not_equal():
    if False:
        for i in range(10):
            print('nop')
    u = URL('http://example.com/path?query=param1#frag')
    bono = URL('http://example.com/path?query=param2#frag')
    assert bono != u

def _test_bad_utf8():
    if False:
        return 10
    bad_bin_url = 'http://xn--9ca.com/%00%FF/%C3%A9'
    url = URL(bad_bin_url)
    expected = 'http://é.com/%00%FF/é'
    actual = url.to_text()
    assert expected == actual

def test_userinfo():
    if False:
        i = 10
        return i + 15
    url = URL('http://someuser:somepassword@example.com/some-segment@ignore')
    assert url.username == 'someuser'
    assert url.password == 'somepassword'
    assert url.to_text() == 'http://someuser:somepassword@example.com/some-segment@ignore'

def test_quoted_userinfo():
    if False:
        return 10
    url = URL('http://wikipedia.org')
    url.username = u'user'
    url.password = u'p@ss'
    assert url.to_text(full_quote=True) == 'http://user:p%40ss@wikipedia.org'
    url = URL(u'http://beyoncé:béb@tmp.com')
    assert url.to_text(full_quote=True) == u'http://beyonc%C3%A9:b%C3%A9b@tmp.com'

def test_mailto():
    if False:
        i = 10
        return i + 15
    mt = 'mailto:mahmoud@hatnote.com'
    url = URL(mt)
    assert url.scheme == 'mailto'
    assert url.to_text() == mt
REL_URL_BASE = 'http://a/b/c/d;p?q'
REL_URL_TEST_CASES = [('g', 'http://a/b/c/g'), ('./g', 'http://a/b/c/g'), ('g/', 'http://a/b/c/g/'), ('/g', 'http://a/g'), (';x', 'http://a/b/c/;x'), ('g;x', 'http://a/b/c/g;x'), ('', 'http://a/b/c/d;p?q'), ('.', 'http://a/b/c/'), ('./', 'http://a/b/c/'), ('..', 'http://a/b/'), ('../', 'http://a/b/'), ('../g', 'http://a/b/g'), ('../..', 'http://a/'), ('../../', 'http://a/'), ('../../g', 'http://a/g'), ('../../../g', 'http://a/g'), ('../../../../g', 'http://a/g'), ('/./g', 'http://a/g'), ('/../g', 'http://a/g'), ('g.', 'http://a/b/c/g.'), ('.g', 'http://a/b/c/.g'), ('g..', 'http://a/b/c/g..'), ('..g', 'http://a/b/c/..g'), ('./../g', 'http://a/b/g'), ('./g/.', 'http://a/b/c/g/'), ('g/./h', 'http://a/b/c/g/h'), ('g/../h', 'http://a/b/c/h'), ('g;x=1/./y', 'http://a/b/c/g;x=1/y'), ('g;x=1/../y', 'http://a/b/c/y')]

def test_rel_navigate():
    if False:
        print('Hello World!')
    for (suffix, expected) in REL_URL_TEST_CASES:
        url = URL(REL_URL_BASE)
        new_url = url.navigate(suffix)
        assert new_url.to_text() == expected
        new_url = url.navigate(URL(suffix))
        assert new_url.to_text() == expected
    return

def test_navigate():
    if False:
        i = 10
        return i + 15
    orig_text = u'http://a.b/c/d?e#f'
    orig = URL(orig_text)
    navd = orig.navigate('')
    assert navd.to_text() == u'http://a.b/c/d?e'
    navd = orig.navigate('dd')
    assert navd.to_text() == u'http://a.b/c/dd'
    navd = orig.navigate('dd/')
    assert navd.to_text() == u'http://a.b/c/dd/'
    navd = orig.navigate('/C')
    assert navd.to_text() == u'http://a.b/C'
    navd = orig.navigate('?e=E&ee=EE')
    assert navd.to_text() == u'http://a.b/c/d?e=E&ee=EE'
    navd = orig.navigate('#FFF')
    assert navd.to_text() == u'http://a.b/c/d?e#FFF'
    navd = orig.navigate('https:')
    assert navd.to_text() == u'https://a.b/c/d?e'
    navd = orig.navigate('//newhost')
    assert navd.to_text() == u'http://newhost/c/d?e'
    _dest_text = u'http://hatnote.com'
    _dest = URL(_dest_text)
    navd = orig.navigate(_dest)
    assert _dest is not navd
    assert navd.to_text() == _dest_text
    navd = orig.navigate(_dest_text)
    assert navd.to_text() == _dest_text

@pytest.mark.parametrize(('expected', 'base', 'paths'), [('https://host/b', 'https://host', ('a', '/b')), ('https://host/b', 'https://host', ('a', 'b')), ('https://host/a/b', 'https://host', ('a/', 'b')), ('https://host/b', 'https://host', ('/a', 'b')), ('https://host/a/b', 'https://host/a/', (None, 'b')), ('https://host/b', 'https://host/a', (None, 'b'))])
def test_chained_navigate(expected, base, paths):
    if False:
        i = 10
        return i + 15
    'Chained :meth:`navigate` calls produces correct results.'
    url = URL(base)
    for path in paths:
        url = url.navigate(path)
    assert expected == url.to_text()

def test_self_normalize():
    if False:
        i = 10
        return i + 15
    url = URL('http://hatnote.com/a/../../b?k=v#hashtags')
    url.normalize()
    assert url.to_text() == 'http://hatnote.com/b?k=v#hashtags'

def test_normalize_with_case():
    if False:
        print('Hello World!')
    url1 = URL('example://a/b/c/%7Bfoo%7D')
    url2 = URL('eXAMPLE://a/./b/../b/%63/%7bfoo%7d')
    assert url1 != url2
    url1.normalize()
    url2.normalize()
    assert url1 == url2

def test_netloc_slashes():
    if False:
        while True:
            i = 10
    url = URL('mailto:mahmoud@hatnote.com')
    assert url.scheme == 'mailto'
    assert url.to_text() == 'mailto:mahmoud@hatnote.com'
    url = URL('http://hatnote.com')
    assert url.scheme == 'http'
    assert url.to_text() == 'http://hatnote.com'
    url = URL('newscheme:a:b:c')
    assert url.scheme == 'newscheme'
    assert url.to_text() == 'newscheme:a:b:c'
    url = URL('newerscheme://a/b/c')
    assert url.scheme == 'newerscheme'
    assert url.to_text() == 'newerscheme://a/b/c'
    url = URL('git+ftp://gitstub.biz/glyph/lefkowitz')
    assert url.scheme == 'git+ftp'
    assert url.to_text() == 'git+ftp://gitstub.biz/glyph/lefkowitz'
    url = URL('what+mailto:freerealestate@enotuniq.org')
    assert url.scheme == 'what+mailto'
    assert url.to_text() == 'what+mailto:freerealestate@enotuniq.org'
    url = URL()
    url.scheme = 'ztp'
    url.path = '/x/y/z'
    assert url.to_text() == 'ztp:/x/y/z'
    url = URL()
    url.scheme = 'git+ftp'
    url.path = '/x/y/z/'
    assert url.to_text() == 'git+ftp:///x/y/z/'
    url = URL('file:///path/to/heck')
    url.scheme = 'mailto'
    assert url.to_text() == 'mailto:/path/to/heck'
    return
FAL_TESTS = [('hello http://world.com/!', ['http://world.com/']), ('hello http://world.com/with?param=true&stuff=yes', ['http://world.com/with?param=true&stuff=yes']), ('http://url.com/w(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', ['http://url.com/w']), ('http://url.com/withmany.......................................', ['http://url.com/withmany']), ('http://url.com/withmany((((((((((((((((((((((((((((((((((a)', ['http://url.com/withmany']), ('http://foo.com/blah_blah', ['http://foo.com/blah_blah']), ('http://foo.com/blah_blah/', ['http://foo.com/blah_blah/']), ('(Something like http://foo.com/blah_blah)', ['http://foo.com/blah_blah']), ('http://foo.com/blah_blah_(wikipedia)', ['http://foo.com/blah_blah_(wikipedia)']), ('http://foo.com/blah_(blah)_(wikipedia)_blah', ['http://foo.com/blah_(blah)_(wikipedia)_blah']), ('http://foo.com/blah_blah.', ['http://foo.com/blah_blah']), ('http://foo.com/blah_blah/.', ['http://foo.com/blah_blah/']), ('<http://foo.com/blah_blah>', ['http://foo.com/blah_blah']), ('<http://foo.com/blah_blah/>', ['http://foo.com/blah_blah/']), ('http://foo.com/blah_blah,', ['http://foo.com/blah_blah']), ('http://www.example.com/wpstyle/?p=364.', ['http://www.example.com/wpstyle/?p=364']), ('rdar://1234', ['rdar://1234']), ('rdar:/1234', ['rdar:/1234']), ('http://userid:password@example.com:8080', ['http://userid:password@example.com:8080']), ('http://userid@example.com', ['http://userid@example.com']), ('message://%3c330e7f8409726r6a4ba78dkf1fd71420c1bf6ff@mail.gmail.com%3e', ['message://%3C330e7f8409726r6a4ba78dkf1fd71420c1bf6ff@mail.gmail.com%3e']), (u'http://➡.ws/䨹', [u'http://➡.ws/䨹']), ('<tag>http://example.com</tag>', ['http://example.com']), ('Just a www.example.com link.', ['https://www.example.com']), ('www.a-link.com', ['https://www.a-link.com']), ('www.a-link.com and www.b-link.com/blogs extra', ['https://www.a-link.com', 'https://www.b-link.com/blogs'])]

def test_find_all_links_basic():
    if False:
        i = 10
        return i + 15
    target = "hi my name is prince nigeria, please visit my website\n    http://richprince.biz or if that's blocked try\n    https://getprince.ly! Thanks for your attention.bye!\n\n    PS if those ports are blocked, how about trying\n    https://crownbux.afamilycompany:broken/affiliate\n\n    PPS if all else fails you can always mailto:thePrince@machovelli.an\n    "
    urls = find_all_links(target)
    assert len(urls) == 2

def test_find_all_links():
    if False:
        i = 10
        return i + 15
    prefix = 'a little something before, '
    suffix = ' a bit of another after.'
    for (content, expected_links) in FAL_TESTS:
        text = prefix + content + suffix
        links = find_all_links(text)
        assert len(links) == len(expected_links)
        for (link, expected) in zip(links, expected_links):
            assert link.to_text(full_quote=False) == expected
        link_tokens = find_all_links(text, with_text=True)
        assert link_tokens[0].startswith(prefix)
        assert link_tokens[-1].endswith(suffix)

def test_unicodey():
    if False:
        i = 10
        return i + 15
    unicodey = u'http://é.com/é?á=í#ú'
    url = URL(unicodey)
    assert url.host == u'é.com'
    assert url.path_parts[1] == u'é'
    assert url.to_text(full_quote=False) == unicodey
    fully_quoted = 'http://xn--9ca.com/%C3%A9?%C3%A1=%C3%AD#%C3%BA'
    assert url.to_text(full_quote=True) == fully_quoted

def test_str_repr():
    if False:
        for i in range(10):
            print('nop')
    assert str(URL('http://googlewebsite.com/e-shops.aspx')) == 'http://googlewebsite.com/e-shops.aspx'