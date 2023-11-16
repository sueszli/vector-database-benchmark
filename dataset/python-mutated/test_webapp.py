from time import sleep
url = 'http://localhost:11111/'

def test_index(browser):
    if False:
        print('Hello World!')
    browser.visit(url)
    assert browser.is_text_present('searxng')

def test_404(browser):
    if False:
        print('Hello World!')
    browser.visit(url + 'missing_link')
    assert browser.is_text_present('Page not found')

def test_about(browser):
    if False:
        return 10
    browser.visit(url)
    browser.links.find_by_text('searxng').click()
    assert browser.is_text_present('Why use it?')

def test_preferences(browser):
    if False:
        i = 10
        return i + 15
    browser.visit(url)
    browser.links.find_by_href('/preferences').click()
    assert browser.is_text_present('Preferences')
    assert browser.is_text_present('COOKIES')
    assert browser.is_element_present_by_xpath('//label[@for="checkbox_dummy"]')

def test_preferences_engine_select(browser):
    if False:
        print('Hello World!')
    browser.visit(url)
    browser.links.find_by_href('/preferences').click()
    assert browser.is_element_present_by_xpath('//label[@for="tab-engines"]')
    browser.find_by_xpath('//label[@for="tab-engines"]').first.click()
    assert not browser.find_by_xpath('//input[@id="engine_general_dummy__general"]').first.checked
    browser.find_by_xpath('//label[@for="engine_general_dummy__general"]').first.check()
    browser.find_by_xpath('//input[@type="submit"]').first.click()
    sleep(1)
    browser.visit(url)
    browser.links.find_by_href('/preferences').click()
    browser.find_by_xpath('//label[@for="tab-engines"]').first.click()
    assert browser.find_by_xpath('//input[@id="engine_general_dummy__general"]').first.checked

def test_preferences_locale(browser):
    if False:
        for i in range(10):
            print('nop')
    browser.visit(url)
    browser.links.find_by_href('/preferences').click()
    browser.find_by_xpath('//label[@for="tab-ui"]').first.click()
    browser.select('locale', 'fr')
    browser.find_by_xpath('//input[@type="submit"]').first.click()
    sleep(1)
    browser.visit(url)
    browser.links.find_by_href('/preferences').click()
    browser.is_text_present('Préférences')

def test_search(browser):
    if False:
        for i in range(10):
            print('nop')
    browser.visit(url)
    browser.fill('q', 'test search query')
    browser.find_by_xpath('//button[@type="submit"]').first.click()
    assert browser.is_text_present('No results were found')