import pytest
from tribler.core.utilities.tracker_utils import MalformedTrackerURLException, get_uniformed_tracker_url, parse_tracker_url
TRACKER_HOST = 'tracker.example.com'
EXPECTED_UNIFORM_URLS = [(f'udp://{TRACKER_HOST}:80', f'udp://{TRACKER_HOST}:80'), (f'udp://{TRACKER_HOST}', None), (f'udp://{TRACKER_HOST}:6969/announce', f'udp://{TRACKER_HOST}:6969'), (f'udp://{TRACKER_HOST}:80\x00', f'udp://{TRACKER_HOST}:80'), (f'udp://{TRACKER_HOST}:80ÿ', None), (f'http://{TRACKER_HOST}:6969/announce', f'http://{TRACKER_HOST}:6969/announce'), (f'http://{TRACKER_HOST}:6969/announce/', f'http://{TRACKER_HOST}:6969/announce'), (f'http://{TRACKER_HOST}', None), (f'http://{TRACKER_HOST}\xa0', None), (f'http://{TRACKER_HOST}/announce', f'http://{TRACKER_HOST}/announce'), (f'http://{TRACKER_HOST}:80/announce', f'http://{TRACKER_HOST}/announce'), (f'http://{TRACKER_HOST}/?do=upload', None), (f'http://{TRACKER_HOST}:80/anno...', None), (f'https://{TRACKER_HOST}/announce', f'https://{TRACKER_HOST}/announce'), (f'https://{TRACKER_HOST}:443/announce', f'https://{TRACKER_HOST}/announce'), (f'unknown://{TRACKER_HOST}/announce', None), (f'wss://{TRACKER_HOST}:80/announce', None), ('ftp://tracker.exampÿle.org:80/announce', None), (';', None), ('', None)]

@pytest.mark.parametrize('given_url, expected_uniform_url', EXPECTED_UNIFORM_URLS)
def test_get_uniformed_tracker_url(given_url, expected_uniform_url):
    if False:
        print('Hello World!')
    uniform_url = get_uniformed_tracker_url(given_url)
    assert uniform_url == expected_uniform_url
PARSED_TRACKER_URLS = [(f'udp://{TRACKER_HOST}:80', ('udp', (f'{TRACKER_HOST}', 80), '')), (f'http://{TRACKER_HOST}:6969/announce', ('http', (f'{TRACKER_HOST}', 6969), '/announce')), (f'https://{TRACKER_HOST}:6969/announce', ('https', (f'{TRACKER_HOST}', 6969), '/announce')), (f'http://{TRACKER_HOST}/announce', ('http', (f'{TRACKER_HOST}', 80), '/announce')), (f'https://{TRACKER_HOST}/announce', ('https', (f'{TRACKER_HOST}', 443), '/announce')), (f'http://ipv6.{TRACKER_HOST}:6969/announce', ('http', (f'ipv6.{TRACKER_HOST}', 6969), '/announce')), (f'https://ipv6.{TRACKER_HOST}:6969/announce', ('https', (f'ipv6.{TRACKER_HOST}', 6969), '/announce'))]

@pytest.mark.parametrize('given_url, expected_parsed_url_tuple', PARSED_TRACKER_URLS)
def test_parse_tracker_url(given_url, expected_parsed_url_tuple):
    if False:
        i = 10
        return i + 15
    parsed_url_tuple = parse_tracker_url(given_url)
    assert parsed_url_tuple == expected_parsed_url_tuple
PARSED_TRACKER_URLS_WITH_FAILURE = [f'unknown://ipv6.{TRACKER_HOST}:6969/announce', f'http://{TRACKER_HOST}:6969/announce( %(', f'https://{TRACKER_HOST}:6969/announce( %(', f'unknown://{TRACKER_HOST}:80', f'http://ipv6.{TRACKER_HOST}:6969', f'https://ipv6.{TRACKER_HOST}:6969', f'udp://{TRACKER_HOST}']

@pytest.mark.parametrize('given_url', PARSED_TRACKER_URLS_WITH_FAILURE)
def test_parse_tracker_url_with_error(given_url):
    if False:
        return 10
    with pytest.raises(MalformedTrackerURLException):
        parse_tracker_url(given_url)