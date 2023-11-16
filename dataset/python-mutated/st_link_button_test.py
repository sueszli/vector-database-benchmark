from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_link_button_display(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        while True:
            i = 10
    'Test that st.link_button renders correctly.'
    link_elements = themed_app.get_by_test_id('stLinkButton')
    expect(link_elements).to_have_count(6)
    for (i, element) in enumerate(link_elements.all()):
        assert_snapshot(element, name=f'link-button-{i}')
        element.hover()
        assert_snapshot(element, name=f'link-button-hover-{i}')
        element.focus()
        assert_snapshot(element, name=f'link-button-focus-{i}')