from hypothesis import given
from hypothesis.strategies import emails, just

@given(emails())
def test_is_valid_email(address: str):
    if False:
        for i in range(10):
            print('nop')
    (local, at_, domain) = address.rpartition('@')
    assert len(address) <= 254
    assert at_ == '@'
    assert local
    assert domain
    assert not domain.lower().endswith('.arpa')

@given(emails(domains=just('mydomain.com')))
def test_can_restrict_email_domains(address: str):
    if False:
        for i in range(10):
            print('nop')
    assert address.endswith('@mydomain.com')