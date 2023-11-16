from sentry.utils.avatar import get_letter_avatar

def test_letter_avatar():
    if False:
        return 10
    letter_avatar = get_letter_avatar('Jane Bloggs', 'janebloggs@example.com')
    assert 'JB' in letter_avatar
    assert '#4e3fb4' in letter_avatar
    assert 'svg' in letter_avatar
    letter_avatar = get_letter_avatar('johnsmith@example.com', 2)
    assert 'J' in letter_avatar
    assert '#57be8c' in letter_avatar
    letter_avatar = get_letter_avatar(None, '127.0.0.1')
    assert '?' in letter_avatar
    assert '#ec5e44' in letter_avatar
    letter_avatar = get_letter_avatar('johnsmith@example.com ', 2)
    assert 'J' in letter_avatar
    assert '#57be8c' in letter_avatar
    letter_avatar = get_letter_avatar('Jane Bloggs', 'janebloggs@example.com', use_svg=False)
    assert 'JB' in letter_avatar
    assert '#4e3fb4' in letter_avatar
    assert 'span' in letter_avatar
    letter_avatar = get_letter_avatar('johnsmith@example.com', 2, use_svg=False)
    assert 'J' in letter_avatar
    assert '#57be8c' in letter_avatar
    letter_avatar = get_letter_avatar(None, '127.0.0.1', use_svg=False)
    assert '?' in letter_avatar
    assert '#ec5e44' in letter_avatar