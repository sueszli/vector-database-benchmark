from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run
EXPANDER_HEADER_IDENTIFIER = '.streamlit-expanderHeader'

def test_default_selection_first_tab(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test st.tabs has first tab selected as default.'
    assert_snapshot(app.locator('.stTabs'), name='tabs-default')

def test_maintains_selection_when_other_tab_added(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test st.tabs maintains selected tab if additional tab added.'
    control_buttons = app.locator('.stButton')
    tab_buttons = app.locator('.stTabs button[role=tab]')
    tab_buttons.nth(1).click()
    control_buttons.nth(0).click()
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    assert_snapshot(app.locator('.stTabs'), name='tabs-selection-add-tab')

def test_maintains_selection_when_other_tab_removed(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Test st.tabs maintains selected tab if non-selected tab removed.'
    control_buttons = app.locator('.stButton')
    control_buttons.nth(5).click()
    wait_for_app_run(app)
    control_buttons.nth(0).click()
    tab_buttons = app.locator('.stTabs button[role=tab]')
    tab_buttons.nth(2).click()
    control_buttons.nth(1).click()
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    assert_snapshot(app.locator('.stTabs'), name='tabs-selection-remove-tab')

def test_resets_selection_when_selected_tab_removed(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        while True:
            i = 10
    'Test st.tabs resets selected tab to 1 if previously selected tab removed.'
    control_buttons = app.locator('.stButton')
    control_buttons.nth(5).click()
    wait_for_app_run(app)
    tab_buttons = app.locator('.stTabs button[role=tab]')
    tab_buttons.nth(1).click()
    control_buttons.nth(2).click()
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    assert_snapshot(app.locator('.stTabs'), name='tabs-remove-selected')

def test_maintains_selection_when_same_name_exists(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test when tabs names change, keeps selected tab if matching label still exists.'
    control_buttons = app.locator('.stButton')
    control_buttons.nth(5).click()
    wait_for_app_run(app)
    control_buttons.nth(0).click()
    tab_buttons = app.locator('.stTabs button[role=tab]')
    tab_buttons.nth(1).click()
    control_buttons.nth(3).click()
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    assert_snapshot(app.locator('.stTabs'), name='tabs-change-some-names')

def test_resets_selection_when_tab_names_change(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test when tabs names change, reset selected tab if no matching label exists.'
    control_buttons = app.locator('.stButton')
    control_buttons.nth(5).click()
    wait_for_app_run(app)
    tab_buttons = app.locator('.stTabs button[role=tab]')
    tab_buttons.nth(1).click()
    control_buttons.nth(4).click()
    wait_for_app_run(app)
    app.wait_for_timeout(1000)
    assert_snapshot(app.locator('.stTabs'), name='tabs-change-all-names')