"""Test how before and after pseudo elements are drawn."""
from ..testing_utils import assert_no_logs

@assert_no_logs
def test_before_after_1(assert_same_renderings):
    if False:
        return 10
    assert_same_renderings('\n            <style>\n                @page { size: 300px 30px }\n                body { margin: 0 }\n                a[href]:before { content: \'[\' attr(href) \'] \' }\n            </style>\n            <p><a href="some url">some content</a></p>\n        ', '\n            <style>\n                @page { size: 300px 30px }\n                body { margin: 0 }\n            </style>\n            <p><a href="another url"><span>[some url] </span>some content</p>\n        ', tolerance=10)

@assert_no_logs
def test_before_after_2(assert_same_renderings):
    if False:
        return 10
    assert_same_renderings("\n            <style>\n                @page { size: 500px 30px }\n                body { margin: 0; quotes: '«' '»' '“' '”' }\n                q:before { content: open-quote '\xa0'}\n                q:after { content: '\xa0' close-quote }\n            </style>\n            <p><q>Lorem ipsum <q>dolor</q> sit amet</q></p>\n        ", '\n            <style>\n                @page { size: 500px 30px }\n                body { margin: 0 }\n                q:before, q:after { content: none }\n            </style>\n            <p><span><span>«\xa0</span>Lorem ipsum\n                <span><span>“\xa0</span>dolor<span>\xa0”</span></span>\n                sit amet<span>\xa0»</span></span></p>\n        ', tolerance=10)

@assert_no_logs
def test_before_after_3(assert_same_renderings):
    if False:
        for i in range(10):
            print('nop')
    assert_same_renderings("\n            <style>\n                @page { size: 100px 30px }\n                body { margin: 0; }\n                p:before { content: 'a' url(pattern.png) 'b'}\n            </style>\n            <p>c</p>\n        ", '\n            <style>\n                @page { size: 100px 30px }\n                body { margin: 0 }\n            </style>\n            <p><span>a<img src="pattern.png" alt="Missing image">b</span>c</p>\n        ', tolerance=10)