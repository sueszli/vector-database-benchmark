import urllib.request
import pytest
try:
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support.expected_conditions import staleness_of, title_is
    from selenium.common.exceptions import NoSuchElementException
except:
    pass

class WaitForPageLoad(object):

    def __init__(self, browser):
        if False:
            for i in range(10):
                print('nop')
        self.browser = browser

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.old_page = self.browser.find_element_by_tag_name('html')

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        WebDriverWait(self.browser, 10).until(staleness_of(self.old_page))

def getContextUrl(browser):
    if False:
        while True:
            i = 10
    return browser.execute_script('return window.location.toString()')

def getUrl(url):
    if False:
        while True:
            i = 10
    content = urllib.request.urlopen(url).read()
    assert 'server error' not in content.lower(), 'Got a server error! ' + repr(url)
    return content

@pytest.mark.usefixtures('resetSettings')
@pytest.mark.webtest
class TestWeb:

    def testFileSecurity(self, site_url):
        if False:
            for i in range(10):
                print('nop')
        assert 'Not Found' in getUrl('%s/media/sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/media/./sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/media/../config.py' % site_url)
        assert 'Forbidden' in getUrl('%s/media/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/media/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/..//sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/media/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../../zeronet.py' % site_url)
        assert 'Not Found' in getUrl('%s/raw/sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/raw/./sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/raw/../config.py' % site_url)
        assert 'Forbidden' in getUrl('%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/..//sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../../zeronet.py' % site_url)
        assert 'Forbidden' in getUrl('%s/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/..//sites.json' % site_url)
        assert 'Forbidden' in getUrl('%s/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/../../zeronet.py' % site_url)
        assert 'Forbidden' in getUrl('%s/content.db' % site_url)
        assert 'Forbidden' in getUrl('%s/./users.json' % site_url)
        assert 'Forbidden' in getUrl('%s/./key-rsa.pem' % site_url)
        assert 'Forbidden' in getUrl('%s/././././././././././//////sites.json' % site_url)

    def testLinkSecurity(self, browser, site_url):
        if False:
            return 10
        browser.get('%s/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/test/security.html' % site_url)
        WebDriverWait(browser, 10).until(title_is('ZeroHello - ZeroNet'))
        assert getContextUrl(browser) == '%s/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/test/security.html' % site_url
        browser.switch_to.frame(browser.find_element_by_id('inner-iframe'))
        assert 'wrapper_nonce' in getContextUrl(browser)
        assert browser.find_element_by_id('script_output').text == 'Result: Works'
        browser.switch_to.default_content()
        browser.switch_to.frame(browser.find_element_by_id('inner-iframe'))
        with WaitForPageLoad(browser):
            browser.find_element_by_id('link_to_current').click()
        assert 'wrapper_nonce' not in getContextUrl(browser)
        assert 'Forbidden' not in browser.page_source
        browser.switch_to.frame(browser.find_element_by_id('inner-iframe'))
        with pytest.raises(NoSuchElementException):
            assert not browser.find_element_by_id('inner-iframe')
        browser.switch_to.default_content()
        browser.switch_to.frame(browser.find_element_by_id('inner-iframe'))
        with WaitForPageLoad(browser):
            browser.find_element_by_id('link_to_top').click()
        assert 'wrapper_nonce' not in getContextUrl(browser)
        assert 'Forbidden' not in browser.page_source
        browser.switch_to.default_content()
        browser.switch_to.frame(browser.find_element_by_id('inner-iframe'))
        assert 'wrapper_nonce' in getContextUrl(browser)
        with WaitForPageLoad(browser):
            browser.execute_script('window.top.location = window.location')
        assert 'wrapper_nonce' in getContextUrl(browser)
        assert '<iframe' in browser.page_source
        browser.switch_to.default_content()

    def testRaw(self, browser, site_url):
        if False:
            print('Hello World!')
        browser.get('%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/test/security.html' % site_url)
        WebDriverWait(browser, 10).until(title_is('Security tests'))
        assert getContextUrl(browser) == '%s/raw/1EU1tbG9oC1A8jz2ouVwGZyQ5asrNsE4Vr/test/security.html' % site_url
        assert browser.find_element_by_id('script_output').text == 'Result: Fail'