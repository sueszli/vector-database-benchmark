import pytest
from pywhat import regex_identifier
from pywhat.filter import Distribution, Filter
from pywhat.helper import load_regexes
database = load_regexes()
r = regex_identifier.RegexIdentifier()
dist = Distribution(Filter({'MinRarity': 0.0}))

def test_regex_successfully_parses():
    if False:
        while True:
            i = 10
    regexes = r.distribution.get_regexes()
    assert type(regexes) == list
    assert len(regexes) != 0
    assert all([type(regex) == dict for regex in regexes])

def regex_valid_match(name: str, match: str) -> bool:
    if False:
        i = 10
        return i + 15
    return any((name in matched['Regex Pattern']['Name'] for matched in r.check([match], dist=dist)))

@pytest.mark.parametrize('name,match', [(regex['Name'], match) for regex in database for match in regex.get('Examples', {}).get('Valid', [])])
def test_regex_valid_match(name: str, match: str):
    if False:
        while True:
            i = 10
    assert regex_valid_match(name, match)

@pytest.mark.parametrize('name,match', [(regex['Name'], match) for regex in database for match in regex.get('Examples', {}).get('Invalid', [])])
def test_regex_invalid_match(name: str, match: str):
    if False:
        return 10
    assert not regex_valid_match(name, match)

@pytest.mark.skip(reason='Fails because not a valid TLD. If presented in punycode, it works.')
def test_international_url():
    if False:
        i = 10
        return i + 15
    assert regex_valid_match('Uniform Resource Locator (URL)', r.check(['http://папироска.рф']))

@pytest.mark.parametrize('match, description', [('00:00:00:00:00:00', 'Xerox Corp'), ('00-00-00-00-00-00', 'Xerox Corp'), ('0000.0000.0000', 'Xerox Corp'), ('5409010000000004', 'UNION NATIONAL BANK'), ('5409 0100 0000 0004', 'UNION NATIONAL BANK'), ('+1-202-555-0156', 'United States'), ('+662025550156', 'Thailand'), ('+356 202 555 0156', 'Malta')])
def test_match_description(match: str, description: str):
    if False:
        for i in range(10):
            print('nop')
    assert description in r.check([match])[0]['Regex Pattern']['Description']

@pytest.mark.parametrize('match, exploit', [('xoxp-514654431830-843187921057-792480346180-d44d2r9b71f954o8z2k5llt41ovpip6v', 'https://slack.com/api/auth.test?token=xoxp-514654431830-843187921057-792480346180-d44d2r9b71f954o8z2k5llt41ovpip6v'), ('xoxb-51465443183-hgvhXVd2ISC2x7gaoRWBOUdQ', 'https://slack.com/api/auth.test?token=xoxb-51465443183-hgvhXVd2ISC2x7gaoRWBOUdQ')])
def test_match_exploit(match: str, exploit: str):
    if False:
        i = 10
        return i + 15
    assert exploit in r.check([match])[0]['Regex Pattern']['Exploit']