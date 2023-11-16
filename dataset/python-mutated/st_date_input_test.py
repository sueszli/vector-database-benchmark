from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_date_input_rendering(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that st.date_input renders correctly via screenshots matching.'
    date_time_widgets = themed_app.get_by_test_id('stDateInput')
    expect(date_time_widgets).to_have_count(14)
    assert_snapshot(date_time_widgets.nth(0), name='date_input-single_date')
    assert_snapshot(date_time_widgets.nth(1), name='date_input-single_datetime')
    assert_snapshot(date_time_widgets.nth(2), name='date_input-range_no_date')
    assert_snapshot(date_time_widgets.nth(3), name='date_input-range_one_date')
    assert_snapshot(date_time_widgets.nth(4), name='date_input-range_two_dates')
    assert_snapshot(date_time_widgets.nth(5), name='date_input-disabled_no_date')
    assert_snapshot(date_time_widgets.nth(6), name='date_input-label_hidden')
    assert_snapshot(date_time_widgets.nth(7), name='date_input-label_collapsed')
    assert_snapshot(date_time_widgets.nth(8), name='date_input-single_date_format')
    assert_snapshot(date_time_widgets.nth(9), name='date_input-range_two_dates_format')
    assert_snapshot(date_time_widgets.nth(10), name='date_input-range_no_date_format')
    assert_snapshot(date_time_widgets.nth(11), name='date_input-single_date_callback')
    assert_snapshot(date_time_widgets.nth(12), name='date_input-empty_value')
    assert_snapshot(date_time_widgets.nth(13), name='date_input-value_from_state')

def test_date_input_has_correct_initial_values(app: Page):
    if False:
        while True:
            i = 10
    'Test that st.date_input has the correct initial values.'
    markdown_elements = app.get_by_test_id('stMarkdown')
    expect(markdown_elements).to_have_count(15)
    expected = ['Value 1: 1970-01-01', 'Value 2: 2019-07-06', 'Value 3: ()', 'Value 4: (datetime.date(2019, 7, 6),)', 'Value 5: (datetime.date(2019, 7, 6), datetime.date(2019, 7, 8))', 'Value 6: ()', 'Value 7: 2019-07-06', 'Value 8: 2019-07-06', 'Value 9: 1970-01-01', 'Value 10: (datetime.date(2019, 7, 6), datetime.date(2019, 7, 8))', 'Value 11: ()', 'Value 12: 1970-01-01', 'Date Input Changed: False', 'Value 13: None', 'Value 14: 1970-02-03']
    for (markdown_element, expected_text) in zip(markdown_elements.all(), expected):
        expect(markdown_element).to_have_text(expected_text, use_inner_text=True)

def test_handles_date_selection(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that selection of a date on the calendar works as expected.'
    app.get_by_test_id('stDateInput').first.click()
    app.locator('[data-baseweb="calendar"] [aria-label^="Choose Friday, January 2nd 1970."]').first.click()
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 1970-01-02', use_inner_text=True)

def test_handle_value_changes(app: Page):
    if False:
        return 10
    'Test that st.date_input has the correct value after typing in a date.'
    first_date_input_field = app.locator('.stDateInput input').first
    first_date_input_field.type('1970/01/02')
    first_date_input_field.blur()
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 1970-01-02', use_inner_text=True)

def test_empty_date_input_behaves_correctly(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test that st.date_input behaves correctly when empty.'
    empty_number_input = app.locator('.stDateInput input').nth(12)
    empty_number_input.type('1970/01/02')
    empty_number_input.press('Enter')
    expect(app.get_by_test_id('stMarkdown').nth(13)).to_have_text('Value 13: 1970-01-02', use_inner_text=True)
    app.get_by_test_id('stMarkdown').nth(13).click()
    assert_snapshot(app.get_by_test_id('stDateInput').nth(12), name='st_date_input-clearable_input')
    empty_number_input = app.locator('.stDateInput input').nth(12)
    empty_number_input.focus()
    empty_number_input.press('Escape')
    app.get_by_test_id('stMarkdown').nth(13).click()
    expect(app.get_by_test_id('stMarkdown').nth(13)).to_have_text('Value 13: None', use_inner_text=True)

def test_handles_range_end_date_changes(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that it correctly handles changes to the end date of a range.'
    app.get_by_test_id('stDateInput').nth(3).click()
    app.locator('[data-baseweb="calendar"] [aria-label^="Choose Wednesday, July 10th 2019."]').first.click()
    expect(app.get_by_test_id('stMarkdown').nth(3)).to_have_text('Value 4: (datetime.date(2019, 7, 6), datetime.date(2019, 7, 10))', use_inner_text=True)

def test_handles_range_start_end_date_changes(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that it correctly handles changes to the start and end date of a range.'
    app.get_by_test_id('stDateInput').nth(4).click()
    app.locator('[data-baseweb="calendar"] [aria-label^="Choose Wednesday, July 10th 2019."]').first.click()
    expect(app.get_by_test_id('stMarkdown').nth(4)).to_have_text('Value 5: (datetime.date(2019, 7, 10),)', use_inner_text=True)
    app.locator('[data-baseweb="calendar"] [aria-label^="Choose Friday, July 12th 2019."]').first.click()
    expect(app.get_by_test_id('stMarkdown').nth(4)).to_have_text('Value 5: (datetime.date(2019, 7, 10), datetime.date(2019, 7, 12))', use_inner_text=True)

def test_calls_callback_on_change(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that it correctly calls the callback on change.'
    app.get_by_test_id('stDateInput').nth(11).click()
    app.locator('[data-baseweb="calendar"] [aria-label^="Choose Friday, January 2nd 1970."]').first.click()
    expect(app.get_by_test_id('stMarkdown').nth(11)).to_have_text('Value 12: 1970-01-02', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(12)).to_have_text('Date Input Changed: True', use_inner_text=True)
    first_date_input_field = app.locator('.stDateInput input').first
    first_date_input_field.type('1971/01/03')
    first_date_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 1971-01-03', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(11)).to_have_text('Value 12: 1970-01-02', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(12)).to_have_text('Date Input Changed: False', use_inner_text=True)

def test_single_date_calendar_picker_rendering(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that the single value calendar picker renders correctly via screenshots matching.'
    themed_app.get_by_test_id('stDateInput').first.click()
    assert_snapshot(themed_app.locator('[data-baseweb="calendar"]').first, name='date_input-single_date_calendar')

def test_range_date_calendar_picker_rendering(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that the range calendar picker renders correctly via screenshots matching.'
    themed_app.get_by_test_id('stDateInput').nth(4).click()
    assert_snapshot(themed_app.locator('[data-baseweb="calendar"]').first, name='date_input-range_two_dates_calendar')

def test_resets_to_default_single_value_if_calendar_closed_empty(app: Page):
    if False:
        return 10
    'Test that single value is reset to default if calendar closed empty.'
    app.get_by_test_id('stDateInput').first.click()
    app.locator('[data-baseweb="calendar"] [aria-label^="Choose Friday, January 2nd 1970."]').first.click()
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 1970-01-02', use_inner_text=True)
    date_input_field = app.locator('.stDateInput input').first
    date_input_field.focus()
    date_input_field.clear()
    app.get_by_test_id('stMarkdown').first.click(delay=100)
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('Value 1: 1970-01-01', use_inner_text=True)

def test_range_is_empty_if_calendar_closed_empty(app: Page):
    if False:
        return 10
    'Test that range value is empty of calendar closed empty.'
    app.get_by_test_id('stDateInput').nth(4).click()
    app.locator('[data-baseweb="calendar"] [aria-label^="Choose Wednesday, July 10th 2019."]').first.click()
    expect(app.get_by_test_id('stMarkdown').nth(4)).to_have_text('Value 5: (datetime.date(2019, 7, 10),)', use_inner_text=True)
    app.locator('[data-baseweb="calendar"] [aria-label^="Choose Friday, July 12th 2019."]').first.click()
    expect(app.get_by_test_id('stMarkdown').nth(4)).to_have_text('Value 5: (datetime.date(2019, 7, 10), datetime.date(2019, 7, 12))', use_inner_text=True)
    date_input_field = app.locator('.stDateInput input').nth(4)
    date_input_field.focus()
    date_input_field.clear()
    app.get_by_test_id('stMarkdown').nth(4).click()
    expect(app.get_by_test_id('stMarkdown').nth(4)).to_have_text('Value 5: ()', use_inner_text=True)