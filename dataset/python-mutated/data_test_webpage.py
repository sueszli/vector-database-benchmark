from httpbin import app

@app.route('/pyspider/test.html')
def test_page():
    if False:
        while True:
            i = 10
    return '\n<a href="/404">404\n<a href="/links/10/0">0\n<a href="/links/10/1">1\n<a href="/links/10/2">2\n<a href="/links/10/3">3\n<a href="/links/10/4">4\n<a href="/gzip">gzip\n<a href="/get">get\n<a href="/deflate">deflate\n<a href="/html">html\n<a href="/xml">xml\n<a href="/robots.txt">robots\n<a href="/cache">cache\n<a href="/stream/20">stream\n'

@app.route('/pyspider/ajax.html')
def test_ajax():
    if False:
        return 10
    return '\n<div class=status>loading...</div>\n<div class=ua></div>\n<div class=ip></div>\n<script>\nvar xhr = new XMLHttpRequest();\nxhr.onload = function() {\n  var data = JSON.parse(xhr.responseText);\n  document.querySelector(\'.status\').innerHTML = \'done\';\n  document.querySelector(\'.ua\').innerHTML = data.headers[\'User-Agent\'];\n  document.querySelector(\'.ip\').innerHTML = data.origin;\n}\nxhr.open("get", "/get", true);\nxhr.send();\n</script>\n'

@app.route('/pyspider/ajax_click.html')
def test_ajax_click():
    if False:
        for i in range(10):
            print('nop')
    return '\n<div class=status>loading...</div>\n<div class=ua></div>\n<div class=ip></div>\n<a href="javascript:void(0)" onclick="load()">load</a>\n<script>\nfunction load() {\n    var xhr = new XMLHttpRequest();\n    xhr.onload = function() {\n      var data = JSON.parse(xhr.responseText);\n      document.querySelector(\'.status\').innerHTML = \'done\';\n      document.querySelector(\'.ua\').innerHTML = data.headers[\'User-Agent\'];\n      document.querySelector(\'.ip\').innerHTML = data.origin;\n    }\n    xhr.open("get", "/get", true);\n    xhr.send();\n}\n</script>\n'