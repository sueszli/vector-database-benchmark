from playwright.sync_api import Page, expect

def test_expandable_state(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test whether expander state is not retained for a distinct expander.'
    app.locator('.stButton button').nth(0).click()
    app.locator("[data-testid='stExpander'] summary").click()
    expect(app.locator("[data-testid='stExpanderDetails']")).to_contain_text('b0_write')
    app.locator('.stButton button').nth(1).click()
    expect(app.locator("[data-testid='stExpanderDetails']")).not_to_contain_text('b0_write')
    expect(app.locator("[data-testid='stExpanderDetails']")).to_be_hidden()