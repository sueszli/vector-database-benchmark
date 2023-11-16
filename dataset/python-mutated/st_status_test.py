from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_status_container_rendering(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Test that st.status renders correctly via screenshots.'
    status_containers = themed_app.get_by_test_id('stExpander')
    expect(status_containers).to_have_count(9)
    assert_snapshot(status_containers.nth(1), name='st_status-complete_state')
    assert_snapshot(status_containers.nth(2), name='st_status-error_state')
    assert_snapshot(status_containers.nth(3), name='st_status-collapsed')
    assert_snapshot(status_containers.nth(4), name='st_status-changed_label')
    assert_snapshot(status_containers.nth(5), name='st_status-without_cm')
    assert_snapshot(status_containers.nth(6), name='st_status-collapsed_via_update')
    assert_snapshot(status_containers.nth(7), name='st_status-empty_state')
    assert_snapshot(status_containers.nth(8), name='st_status-uncaught_exception')

def test_running_state(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that st.status renders a spinner when in running state.'
    running_status = app.get_by_test_id('stExpander').nth(0)
    expect(running_status.get_by_test_id('stExpanderIconSpinner')).to_be_visible()

def test_status_collapses_and_expands(app: Page):
    if False:
        while True:
            i = 10
    'Test that a status collapses and expands.'
    expander_content = 'Doing some work...'
    running_status = app.get_by_test_id('stExpander').nth(0)
    expect(running_status.get_by_text(expander_content)).to_be_visible()
    expander_header = running_status.locator('summary')
    expander_header.click()
    expect(running_status.get_by_text(expander_content)).not_to_be_visible()
    expander_header.click()
    expect(running_status.get_by_text(expander_content)).to_be_visible()