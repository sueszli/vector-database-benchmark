import pytest
from django.template import engines
from django.test import RequestFactory

@pytest.mark.parametrize('input, output', (('{% script %}alert("hi"){% endscript %}', '<script nonce="r@nD0m">alert("hi")</script>'), ('{% script async=True defer=True type="text/javascript" %}alert("hi"){% endscript %}', '<script async defer nonce="r@nD0m" type="text/javascript">alert("hi")</script>'), ('\n        {% script async=True defer=True type="text/javascript" %}\n        <script>alert("hi")</script>\n        {% endscript %}', '<script async defer nonce="r@nD0m" type="text/javascript">alert("hi")</script>'), ('\n        {% script %}\n        <script>\n        alert("hi")\n        </script>\n        {% endscript %}', '<script nonce="r@nD0m">alert("hi")</script>'), ('{% script src="/app.js" %}{% endscript %}', '<script nonce="r@nD0m" src="/app.js"></script>'), ('{% script src=url_path %}{% endscript %}', '<script nonce="r@nD0m" src="/asset.js"></script>'), ('{% script src=url_path|upper %}{% endscript %}', '<script nonce="r@nD0m" src="/ASSET.JS"></script>')))
def test_script_context(input, output):
    if False:
        for i in range(10):
            print('nop')
    request = RequestFactory().get('/')
    request.csp_nonce = 'r@nD0m'
    prefix = '{% load sentry_assets %}'
    result = engines['django'].from_string(prefix + input).render(context={'request': request, 'url_path': '/asset.js'}).strip()
    assert result == output