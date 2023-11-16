import base64
import hashlib
import google
from google.cloud import webrisk_v1
from .compute_threatlist_diff import compute_threatlist_diff
from .search_hashes import search_hashes
from .search_uri import search_uri
from .submit_uri import submit_uri
PROJECT = google.auth.default()[1]

def test_search_uri_with_threat() -> None:
    if False:
        while True:
            i = 10
    response = search_uri('http://testsafebrowsing.appspot.com/s/malware.html', webrisk_v1.ThreatType.MALWARE)
    assert response.threat.threat_types

def test_search_uri_without_threat() -> None:
    if False:
        print('Hello World!')
    response = search_uri('http://testsafebrowsing.appspot.com/malware.html', webrisk_v1.ThreatType.MALWARE)
    assert not response.threat.threat_types

def test_submit_uri() -> None:
    if False:
        for i in range(10):
            print('nop')
    malware_uri = 'http://testsafebrowsing.appspot.com/s/malware.html'
    response = submit_uri(PROJECT, malware_uri)
    assert response.uri == malware_uri

def test_search_hashes() -> None:
    if False:
        print('Hello World!')
    uri = 'http://example.com'
    sha256 = hashlib.sha256()
    sha256.update(base64.urlsafe_b64encode(bytes(uri, 'utf-8')))
    hex_string = sha256.digest()
    search_hashes(hex_string, webrisk_v1.ThreatType.MALWARE)

def test_compute_threatdiff_list() -> None:
    if False:
        print('Hello World!')
    response = compute_threatlist_diff(webrisk_v1.ThreatType.MALWARE, b'', 1024, 1024, webrisk_v1.CompressionType.RAW)
    assert response.response_type