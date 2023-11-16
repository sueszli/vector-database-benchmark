from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_time_input_widget_rendering(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that the time input widgets are correctly rendered via screenshot matching.'
    time_input_widgets = themed_app.get_by_test_id('stTimeInput')
    expect(time_input_widgets).to_have_count(9)
    assert_snapshot(time_input_widgets.nth(0), name='st_time_input-8_45')
    assert_snapshot(time_input_widgets.nth(1), name='st_time_input-21_15_help')
    assert_snapshot(time_input_widgets.nth(2), name='st_time_input-disabled')
    assert_snapshot(time_input_widgets.nth(3), name='st_time_input-hidden_label')
    assert_snapshot(time_input_widgets.nth(4), name='st_time_input-collapsed_label')
    assert_snapshot(time_input_widgets.nth(5), name='st_time_input-callback')
    assert_snapshot(time_input_widgets.nth(6), name='st_time_input-step_60')
    assert_snapshot(time_input_widgets.nth(7), name='st_time_input-empty')
    assert_snapshot(time_input_widgets.nth(8), name='st_time_input-value_from_state')

def test_time_input_has_correct_initial_values(app: Page):
    if False:
        while True:
            i = 10
    'Test that st.time_input returns the correct initial values.'
    markdown_elements = app.get_by_test_id('stMarkdown')
    expect(markdown_elements).to_have_count(10)
    expected = ['Value 1: 08:45:00', 'Value 2: 21:15:00', 'Value 3: 08:45:00', 'Value 4: 08:45:00', 'Value 5: 08:45:00', 'Value 6: 08:45:00', 'time input changed: False', 'Value 7: 08:45:00', 'Value 8: None', 'Value 9: 08:50:00']
    for (markdown_element, expected_text) in zip(markdown_elements.all(), expected):
        expect(markdown_element).to_have_text(expected_text, use_inner_text=True)

def test_handles_time_selection(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that selection of a time via the dropdown works correctly.'
    app.get_by_test_id('stTimeInput').nth(0).locator('input').click()
    selection_dropdown = app.locator('[data-baseweb="popover"]').first
    assert_snapshot(selection_dropdown, name='st_time_input-selection_dropdown')
    selection_dropdown.locator('li').nth(0).click()
    expect(app.get_by_test_id('stMarkdown').nth(0)).to_have_text('Value 1: 00:00:00', use_inner_text=True)

def test_handles_step_correctly(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that the step parameter is correctly applied.'
    app.get_by_test_id('stTimeInput').nth(6).locator('input').click()
    selection_dropdown = app.locator('[data-baseweb="popover"]').first
    assert_snapshot(selection_dropdown, name='st_time_input-step_60_dropdown')
    selection_dropdown.locator('li').nth(1).click()
    expect(app.get_by_test_id('stMarkdown').nth(7)).to_have_text('Value 7: 00:01:00', use_inner_text=True)

def test_handles_time_selection_via_typing(app: Page):
    if False:
        while True:
            i = 10
    'Test that selection of a time via typing works correctly.'
    time_input_field = app.get_by_test_id('stTimeInput').first.locator('input')
    time_input_field.type('00:15')
    time_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 00:15:00', use_inner_text=True)
    time_input_field.type('00:16')
    time_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 00:16:00', use_inner_text=True)

def test_empty_time_input_behaves_correctly(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that st.time_input behaves correctly when empty (no initial value).'
    empty_time_input = app.get_by_test_id('stTimeInput').nth(7)
    empty_time_input_field = empty_time_input.locator('input')
    empty_time_input_field.type('00:15')
    empty_time_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').nth(8)).to_have_text('Value 8: 00:15:00', use_inner_text=True)
    assert_snapshot(empty_time_input, name='st_time_input-clearable_input')
    empty_time_input.get_by_test_id('stTimeInputClearButton').click()
    expect(app.get_by_test_id('stMarkdown').nth(8)).to_have_text('Value 8: None', use_inner_text=True)

def test_keeps_value_on_selection_close(app: Page):
    if False:
        while True:
            i = 10
    'Test that the selection is kept when the dropdown is closed.'
    app.get_by_test_id('stTimeInput').first.locator('input').click()
    expect(app.locator('[data-baseweb="popover"]').first).to_be_visible()
    app.get_by_test_id('stMarkdown').first.click()
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 08:45:00', use_inner_text=True)

def test_handles_callback_on_change_correctly(app: Page):
    if False:
        return 10
    'Test that it correctly calls the callback on change.'
    expect(app.get_by_test_id('stMarkdown').nth(5)).to_have_text('Value 6: 08:45:00', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(6)).to_have_text('time input changed: False', use_inner_text=True)
    app.get_by_test_id('stTimeInput').nth(5).locator('input').click()
    time_dropdown = app.locator('[data-baseweb="popover"]').first
    time_dropdown.locator('li').first.click()
    expect(app.get_by_test_id('stMarkdown').nth(5)).to_have_text('Value 6: 00:00:00', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(6)).to_have_text('time input changed: True', use_inner_text=True)
    empty_time_input_field = app.get_by_test_id('stTimeInput').locator('input').first
    empty_time_input_field.type('00:15')
    empty_time_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 00:15:00', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(5)).to_have_text('Value 6: 00:00:00', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(6)).to_have_text('time input changed: False', use_inner_text=True)