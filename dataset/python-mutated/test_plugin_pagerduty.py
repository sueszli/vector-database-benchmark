from apprise.plugins.NotifyPagerDuty import NotifyPagerDuty
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
apprise_url_tests = (('pagerduty://', {'instance': TypeError}), ('pagerduty://%20@%20/', {'instance': TypeError}), ('pagerduty://%20/', {'instance': TypeError}), ('pagerduty://%20@abcd/', {'instance': TypeError}), ('pagerduty://myroutekey@myapikey/%20', {'instance': TypeError}), ('pagerduty://myroutekey@myapikey/mysource/%20', {'instance': TypeError}), ('pagerduty://myroutekey@myapikey?region=invalid', {'instance': TypeError}), ('pagerduty://myroutekey@myapikey?severity=invalid', {'instance': TypeError}), ('pagerduty://myroutekey@myapikey', {'instance': NotifyPagerDuty, 'privacy_url': 'pagerduty://****@****/A...e/N...n?'}), ('pagerduty://myroutekey@myapikey?image=no', {'instance': NotifyPagerDuty}), ('pagerduty://myroutekey@myapikey?region=eu', {'instance': NotifyPagerDuty}), ('pagerduty://myroutekey@myapikey?severity=critical', {'instance': NotifyPagerDuty}), ('pagerduty://myroutekey@myapikey?severity=err', {'instance': NotifyPagerDuty}), ('pagerduty://myroutekey@myapikey?+key=value&+key2=value2', {'instance': NotifyPagerDuty}), ('pagerduty://myroutekey@myapikey/mysource/mycomponent', {'instance': NotifyPagerDuty, 'privacy_url': 'pagerduty://****@****/m...e/m...t?'}), ('pagerduty://routekey@apikey/ms/mc?group=mygroup&class=myclass', {'instance': NotifyPagerDuty}), ('pagerduty://?integrationkey=r&apikey=a&source=s&component=c&group=g&class=c&image=no&click=http://localhost', {'instance': NotifyPagerDuty}), ('pagerduty://somerkey@someapikey/bizzare/code', {'instance': NotifyPagerDuty, 'response': False, 'requests_response_code': 999}), ('pagerduty://myroutekey@myapikey/mysource/mycomponent', {'instance': NotifyPagerDuty, 'test_requests_exceptions': True}))

def test_plugin_pagerduty_urls():
    if False:
        i = 10
        return i + 15
    '\n    NotifyPagerDuty() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()