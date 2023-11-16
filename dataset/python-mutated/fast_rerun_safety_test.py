from playwright.sync_api import Page, expect

def test_no_concurrent_changes(app: Page):
    if False:
        for i in range(10):
            print('nop')
    counters = app.locator('.stMarkdown')
    expect(counters.first).to_have_text('0', use_inner_text=True)
    button = app.locator('.stButton')
    button.first.click()
    app.wait_for_timeout(300)
    counters = app.locator('.stMarkdown')
    c1 = counters.nth(0).inner_text()
    c2 = counters.nth(1).inner_text()
    assert c1 == c2