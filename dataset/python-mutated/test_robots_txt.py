def test_robots_txt(app):
    if False:
        while True:
            i = 10
    res = app.get(u'/robots.txt')
    assert res.status_code == 200
    assert res.headers.get(u'Content-Type') == u'text/plain; charset=utf-8'
    assert 'User-agent' in res.body