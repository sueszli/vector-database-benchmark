from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run
EXPANDER_HEADER_IDENTIFIER = 'summary'

def test_displays_expander_and_regular_containers_properly(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that expanders and regular containers are displayed properly.'
    main_expanders = app.locator(".main [data-testid='stExpander']")
    expect(main_expanders).to_have_count(3)
    for expander in main_expanders.all():
        expect(expander.locator(EXPANDER_HEADER_IDENTIFIER)).to_be_visible()
    sidebar_expander = app.locator("[data-testid='stSidebar'] [data-testid='stExpander']").first
    expect(sidebar_expander.locator(EXPANDER_HEADER_IDENTIFIER)).to_be_visible()

def test_expander_displays_correctly(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that sidebar and main container expanders are displayed correctly.'
    themed_app.locator('.stButton button').first.focus()
    assert_snapshot(themed_app.locator('.main'), name='expanders-in-main')
    assert_snapshot(themed_app.locator("[data-testid='stSidebar']"), name='expanders-in-sidebar')

def test_expander_collapses_and_expands(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that an expander collapses and expands.'
    main_expanders = app.locator(".main [data-testid='stExpander']")
    expect(main_expanders).to_have_count(3)
    expanders = main_expanders.all()
    expander_header = expanders[0].locator(EXPANDER_HEADER_IDENTIFIER)
    expect(expander_header).to_be_visible()
    toggle = expander_header.locator('svg').first
    expect(toggle).to_be_visible()
    expander_header.click()
    toggle = expander_header.locator('svg').first
    expect(toggle).to_be_visible()
    expander_header = expanders[1].locator(EXPANDER_HEADER_IDENTIFIER)
    expect(expander_header).to_be_visible()
    toggle = expander_header.locator('svg').first
    expect(toggle).to_be_visible()
    expander_header.click()
    toggle = expander_header.locator('svg').first
    expect(toggle).to_be_visible()

def test_empty_expander_not_rendered(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that an empty expander is not rendered.'
    expect(app.get_by_text('Empty expander')).not_to_be_attached()

def test_expander_session_state_set(app: Page):
    if False:
        print('Hello World!')
    'Test that session state updates are propagated to expander content'
    main_expanders = app.locator(".main [data-testid='stExpander']")
    expect(main_expanders).to_have_count(3)
    num_input = main_expanders.nth(2).locator('.stNumberInput input')
    num_input.fill('10')
    num_input.press('Enter')
    main_expanders.nth(2).locator(EXPANDER_HEADER_IDENTIFIER).click()
    app.get_by_text('Update Num Input').click()
    wait_for_app_run(app)
    app.get_by_text('Print State Value').click()
    wait_for_app_run(app)
    text_elements = app.locator("[data-testid='stText']")
    expect(text_elements).to_have_count(2)
    text_elements = text_elements.all_inner_texts()
    texts = [text.strip() for text in text_elements]
    expected = ['0.0', '0.0']
    assert texts == expected