import datetime
import pytest
from django.template import engines
from sentry.models.organization import Organization
from sentry.testutils.helpers.features import Feature

def test_system_origin():
    if False:
        return 10
    result = engines['django'].from_string('\n        {% load sentry_helpers %}\n        {% system_origin %}\n    ').render().strip()
    assert result == 'http://testserver'

@pytest.mark.parametrize('input,output', (('{% absolute_uri %}', 'http://testserver'), ("{% absolute_uri '/matt/' %}", 'http://testserver/matt/'), ("{% absolute_uri '/{}/' 'matt' %}", 'http://testserver/matt/'), ("{% absolute_uri '/{}/' who %}", 'http://testserver/matt/'), ("{% absolute_uri '/foo/x{}/' xxx %}", 'http://testserver/foo/x/'), ("{% absolute_uri '/{}/{}/' who desc %}", 'http://testserver/matt/awesome/'), ('{% absolute_uri as uri %}hello {{ uri }}!', 'hello http://testserver!'), ("{% absolute_uri '/matt/' as uri %}hello {{ uri }}!", 'hello http://testserver/matt/!'), ("{% absolute_uri '/{}/' 'matt' as uri %}hello {{ uri }}!", 'hello http://testserver/matt/!'), ("{% absolute_uri '/{}/' who as uri %}hello {{ uri }}!", 'hello http://testserver/matt/!'), ("{% absolute_uri '/{}/{}/x{}/{}/' who 'xxx' nope desc as uri %}hello {{ uri }}!", 'hello http://testserver/matt/xxx/x/awesome/!')))
def test_absolute_uri(input, output):
    if False:
        while True:
            i = 10
    prefix = '{% load sentry_helpers %}'
    result = engines['django'].from_string(prefix + input).render(context={'who': 'matt', 'desc': 'awesome'}).strip()
    assert result == output

@pytest.mark.parametrize('input,output', (("{% org_url organization '/issues/' %}", 'http://testserver/issues/'), ("{% org_url organization '/issues/' query='referrer=alert' %}", 'http://testserver/issues/?referrer=alert'), ("{% org_url organization '/issues/' query='referrer=alert' fragment='test' %}", 'http://testserver/issues/?referrer=alert#test'), ('{% org_url organization path %}', 'http://testserver/organizations/sentry/issues/')))
def test_org_url(input, output):
    if False:
        i = 10
        return i + 15
    prefix = '{% load sentry_helpers %}'
    org = Organization(id=1, slug='sentry', name='Sentry')
    result = engines['django'].from_string(prefix + input).render(context={'organization': org, 'path': '/organizations/sentry/issues/'}).strip()
    assert result == output

@pytest.mark.parametrize('input,output', (("{% org_url organization '/organizations/sentry/discover/' %}", 'http://sentry.testserver/discover/'), ("{% org_url organization path query='referrer=alert' %}", 'http://sentry.testserver/issues/?referrer=alert')))
def test_org_url_customer_domains(input, output):
    if False:
        return 10
    prefix = '{% load sentry_helpers %}'
    org = Organization(id=1, slug='sentry', name='Sentry')
    with Feature('organizations:customer-domains'):
        result = engines['django'].from_string(prefix + input).render(context={'organization': org, 'path': '/organizations/sentry/issues/'}).strip()
        assert result == output

def test_querystring():
    if False:
        print('Hello World!')
    input = '\n    {% load sentry_helpers %}\n    {% querystring transaction="testing" referrer="weekly_report" space="some thing"%}\n    '
    result = engines['django'].from_string(input).render(context={}).strip()
    assert result == 'transaction=testing&amp;referrer=weekly_report&amp;space=some+thing'

def test_date_handle_date_and_datetime():
    if False:
        return 10
    result = engines['django'].from_string('\n{% load sentry_helpers %}\n{{ date_obj|date:"Y-m-d" }}\n{{ datetime_obj|date:"Y-m-d" }}\n            ').render(context={'date_obj': datetime.date(2021, 4, 16), 'datetime_obj': datetime.datetime(2021, 4, 17, 12, 13, 14)}).strip()
    assert result == '\n'.join(['2021-04-16', '2021-04-17'])

@pytest.mark.parametrize('a_dict,key,expected', (({}, '', ''), ({}, 'hi', ''), ({'hello': 1}, 'hello', '1')))
def test_get_item(a_dict, key, expected):
    if False:
        return 10
    prefix = '{% load sentry_helpers %} {{ something|get_item:"' + key + '" }}'
    result = engines['django'].from_string(prefix).render(context={'something': a_dict}).strip()
    assert result == expected