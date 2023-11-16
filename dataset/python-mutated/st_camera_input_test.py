import pytest
from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run

@pytest.mark.skip_browser('webkit')
def test_displays_correct_number_of_elements(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that it renders correct number of camera_input elements.'
    camera_input_widgets = app.get_by_test_id('stCameraInput')
    expect(camera_input_widgets).to_have_count(2)

@pytest.mark.only_browser('chromium')
def test_captures_photo(app: Page):
    if False:
        for i in range(10):
            print('nop')
    "Test camera_input captures photo when 'Take photo' button clicked."
    app.wait_for_timeout(3000)
    take_photo_button = app.get_by_test_id('stCameraInputButton').first
    take_photo_button.click()
    wait_for_app_run(app, wait_delay=3000)
    expect(app.get_by_test_id('stImage')).to_have_count(1)

@pytest.mark.only_browser('chromium')
def test_clear_photo(app: Page):
    if False:
        while True:
            i = 10
    "Test camera_input removes photo when 'Clear photo' button clicked."
    app.wait_for_timeout(3000)
    take_photo_button = app.get_by_test_id('stCameraInputButton').first
    take_photo_button.click()
    wait_for_app_run(app, wait_delay=3000)
    expect(app.get_by_test_id('stImage')).to_have_count(1)
    remove_photo_button = app.get_by_text('Clear photo').first
    remove_photo_button.click()
    wait_for_app_run(app, wait_delay=3000)
    expect(app.get_by_test_id('stImage')).to_have_count(0)

@pytest.mark.skip_browser('webkit')
def test_shows_disabled_widget_correctly(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Test that it renders disabled camera_input widget correctly.'
    camera_input_widgets = themed_app.get_by_test_id('stCameraInput')
    expect(camera_input_widgets).to_have_count(2)
    disabled_camera_input = camera_input_widgets.nth(1)
    assert_snapshot(disabled_camera_input, name='disabled-camera-input')