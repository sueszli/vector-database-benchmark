from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_main_menu_images(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    themed_app.get_by_test_id('stMainMenu').click()
    element = themed_app.get_by_test_id('stMainMenuPopover')
    assert_snapshot(element, name='main_menu')