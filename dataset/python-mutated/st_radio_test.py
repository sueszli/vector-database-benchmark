from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run

def test_radio_widget_rendering(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test that the radio widgets are correctly rendered via screenshot matching.'
    radio_widgets = themed_app.get_by_test_id('stRadio')
    expect(radio_widgets).to_have_count(13)
    assert_snapshot(radio_widgets.nth(0), name='st_radio-default')
    assert_snapshot(radio_widgets.nth(1), name='st_radio-formatted_options')
    assert_snapshot(radio_widgets.nth(2), name='st_radio-no_options')
    assert_snapshot(radio_widgets.nth(3), name='st_radio-disabled')
    assert_snapshot(radio_widgets.nth(4), name='st_radio-horizontal')
    assert_snapshot(radio_widgets.nth(5), name='st_radio-dataframe_options')
    assert_snapshot(radio_widgets.nth(6), name='st_radio-hidden_label')
    assert_snapshot(radio_widgets.nth(7), name='st_radio-collapsed_label')
    assert_snapshot(radio_widgets.nth(8), name='st_radio-markdown_options')
    assert_snapshot(radio_widgets.nth(9), name='st_radio-captions')
    assert_snapshot(radio_widgets.nth(10), name='st_radio-horizontal_captions')
    assert_snapshot(radio_widgets.nth(11), name='st_radio-callback_help')
    assert_snapshot(radio_widgets.nth(12), name='st_radio-empty_selection')

def test_radio_has_correct_default_values(app: Page):
    if False:
        return 10
    'Test that st.radio returns the correct initial values.'
    markdown_elements = app.get_by_test_id('stMarkdown')
    expect(markdown_elements).to_have_count(14)
    expected = ['value 1: female', 'value 2: male', 'value 3: None', 'value 4: female', 'value 5: female', 'value 6: female', 'value 7: female', 'value 8: female', 'value 9: bold text', 'value 10: A', 'value 11: yes', 'value 12: male', 'radio changed: False', 'value 13: None']
    for (markdown_element, expected_text) in zip(markdown_elements.all(), expected):
        expect(markdown_element).to_have_text(expected_text, use_inner_text=True)

def test_set_value_correctly_when_click(app: Page):
    if False:
        print('Hello World!')
    'Test that st.radio returns the correct values when the selection is changed.'
    for (index, element) in enumerate(app.get_by_test_id('stRadio').all()):
        if index not in [2, 3]:
            element.locator('label[data-baseweb="radio"]').nth(1).click(force=True)
            wait_for_app_run(app)
    expected = ['value 1: male', 'value 2: male', 'value 3: None', 'value 4: female', 'value 5: male', 'value 6: male', 'value 7: male', 'value 8: male', 'value 9: italics text', 'value 10: B', 'value 11: maybe', 'value 12: male', 'radio changed: False', 'value 13: male']
    for (markdown_element, expected_text) in zip(app.get_by_test_id('stMarkdown').all(), expected):
        expect(markdown_element).to_have_text(expected_text, use_inner_text=True)

def test_calls_callback_on_change(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that it correctly calls the callback on change.'
    radio_widget = app.get_by_test_id('stRadio').nth(11)
    radio_widget.locator('label[data-baseweb="radio"]').first.click(force=True)
    wait_for_app_run(app)
    expect(app.get_by_test_id('stMarkdown').nth(11)).to_have_text('value 12: female', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(12)).to_have_text('radio changed: True', use_inner_text=True)
    first_date_input_field = app.get_by_test_id('stRadio').first
    first_date_input_field.locator('label[data-baseweb="radio"]').last.click(force=True)
    wait_for_app_run(app)
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('value 1: male', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(11)).to_have_text('value 12: female', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(12)).to_have_text('radio changed: False', use_inner_text=True)