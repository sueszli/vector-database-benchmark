from functools import partial
import pretend
import pytest
from warehouse.filters import _camo_url
from warehouse.utils.gravatar import gravatar, profile

@pytest.mark.parametrize(('email', 'size', 'expected'), [(None, None, 'https://camo.example.net/11c994d69863c3e9cc62a7467a79cc77daff5cad/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f64343164386364393866303062323034653938303039393865636638343237653f73697a653d3830'), (None, 50, 'https://camo.example.net/5d174a434e25f7c26372e241c7aad866d6ed45e1/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f64343164386364393866303062323034653938303039393865636638343237653f73697a653d3530'), ('', None, 'https://camo.example.net/11c994d69863c3e9cc62a7467a79cc77daff5cad/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f64343164386364393866303062323034653938303039393865636638343237653f73697a653d3830'), ('', 40, 'https://camo.example.net/e65ad014ae9afac08674c0b2201eb9fc52e944bc/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f64343164386364393866303062323034653938303039393865636638343237653f73697a653d3430'), ('foo@example.com', None, 'https://camo.example.net/32bd80f8aab1ac713fba756662716138ba97a75b/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f62343864656636343537353862393535333764343432346338346431613966663f73697a653d3830'), ('foo@example.com', 100, 'https://camo.example.net/1d9822f304be5013ba8fef23dc5849fb3066ac66/68747470733a2f2f7365637572652e67726176617461722e636f6d2f6176617461722f62343864656636343537353862393535333764343432346338346431613966663f73697a653d313030')])
def test_gravatar(email, size, expected, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    request = pretend.stub(registry=pretend.stub(settings={'camo.url': 'https://camo.example.net/', 'camo.key': 'fake key'}))
    camo_url = partial(_camo_url, request)
    request.camo_url = camo_url
    kwargs = {}
    if size is not None:
        kwargs['size'] = size
    assert gravatar(request, email, **kwargs) == expected

def test_profile():
    if False:
        i = 10
        return i + 15
    email = 'foo@example.com'
    expected = 'https://gravatar.com/b48def645758b95537d4424c84d1a9ff'
    assert profile(email) == expected