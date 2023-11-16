from playwright.sync_api import Page, expect

def test_nested_expanders(app: Page):
    if False:
        return 10
    'Test that st.expander may not be nested inside other expanders.'
    exception_message = app.locator('.stException .message')
    expect(exception_message).to_have_text('StreamlitAPIException: Expanders may not be nested inside other expanders.')