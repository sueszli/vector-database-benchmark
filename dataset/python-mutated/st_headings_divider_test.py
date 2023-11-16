from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_header_display(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that st.header renders correctly with dividers.'
    header_elements = app.locator('.stHeadingContainer')
    expect(header_elements).to_have_count(16)
    for (i, element) in enumerate(header_elements.all()):
        if i < 8:
            assert_snapshot(element, name=f'header-divider-{i}')

def test_subheader_display(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that st.subheader renders correctly with dividers.'
    subheader_elements = app.locator('.stHeadingContainer')
    expect(subheader_elements).to_have_count(16)
    for (i, element) in enumerate(subheader_elements.all()):
        if i > 7:
            assert_snapshot(element, name=f'subheader-divider-{i}')