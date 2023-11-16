from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_text_input_widget_rendering(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that the st.text_input widgets are correctly rendered via screenshot matching.'
    text_input_widgets = themed_app.get_by_test_id('stTextInput')
    expect(text_input_widgets).to_have_count(11)
    assert_snapshot(text_input_widgets.nth(0), name='st_text_input-default')
    assert_snapshot(text_input_widgets.nth(1), name='st_text_input-value_some_text')
    assert_snapshot(text_input_widgets.nth(2), name='st_text_input-value_1234')
    assert_snapshot(text_input_widgets.nth(3), name='st_text_input-value_None')
    assert_snapshot(text_input_widgets.nth(4), name='st_text_input-placeholder')
    assert_snapshot(text_input_widgets.nth(5), name='st_text_input-disabled')
    assert_snapshot(text_input_widgets.nth(6), name='st_text_input-hidden_label')
    assert_snapshot(text_input_widgets.nth(7), name='st_text_input-collapsed_label')
    assert_snapshot(text_input_widgets.nth(8), name='st_text_input-callback_help')
    assert_snapshot(text_input_widgets.nth(9), name='st_text_input-max_chars_5')
    assert_snapshot(text_input_widgets.nth(10), name='st_text_input-type_password')

def test_text_input_has_correct_initial_values(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that st.text_input has the correct initial values.'
    markdown_elements = app.get_by_test_id('stMarkdown')
    expect(markdown_elements).to_have_count(12)
    expected = ['value 1: ', 'value 2: some text', 'value 3: 1234', 'value 4: None', 'value 5: ', 'value 6: default text', 'value 7: default text', 'value 8: default text', 'value 9: ', 'text input changed: False', 'value 10: 1234', 'value 11: my password']
    for (markdown_element, expected_text) in zip(markdown_elements.all(), expected):
        expect(markdown_element).to_have_text(expected_text, use_inner_text=True)

def test_text_input_shows_instructions_when_dirty(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that st.text_input shows the instructions correctly when dirty.'
    text_input = app.get_by_test_id('stTextInput').nth(9)
    text_input_field = text_input.locator('input').first
    text_input_field.fill('123')
    assert_snapshot(text_input, name='st_text_input-input_instructions')

def test_text_input_limits_input_via_max_chars(app: Page):
    if False:
        return 10
    'Test that st.text_input correctly limits the number of characters via max_chars.'
    text_input_field = app.get_by_test_id('stTextInput').nth(9).locator('input').first
    text_input_field.clear()
    text_input_field.type('12345678')
    text_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').nth(10)).to_have_text('value 10: 12345', use_inner_text=True)
    text_input_field.focus()
    text_input_field.fill('12345678')
    text_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').nth(10)).to_have_text('value 10: 12345', use_inner_text=True)

def test_text_input_has_correct_value_on_blur(app: Page):
    if False:
        while True:
            i = 10
    'Test that st.text_input has the correct value on blur.'
    first_text_input_field = app.get_by_test_id('stTextInput').first.locator('input').first
    first_text_input_field.focus()
    first_text_input_field.fill('hello world')
    first_text_input_field.blur()
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('value 1: hello world', use_inner_text=True)

def test_text_input_has_correct_value_on_enter(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that st.text_input has the correct value on enter.'
    first_text_input_field = app.get_by_test_id('stTextInput').first.locator('input').first
    first_text_input_field.focus()
    first_text_input_field.fill('hello world')
    first_text_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('value 1: hello world', use_inner_text=True)

def test_text_input_has_correct_value_on_click_outside(app: Page):
    if False:
        while True:
            i = 10
    'Test that st.text_input has the correct value on click outside.'
    first_text_input_field = app.get_by_test_id('stTextInput').first.locator('input').first
    first_text_input_field.focus()
    first_text_input_field.fill('hello world')
    app.get_by_test_id('stMarkdown').first.click()
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('value 1: hello world', use_inner_text=True)

def test_empty_text_input_behaves_correctly(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that st.text_input behaves correctly when empty.'
    expect(app.get_by_test_id('stMarkdown').nth(3)).to_have_text('value 4: None', use_inner_text=True)
    empty_text_input = app.get_by_test_id('stTextInput').nth(3)
    empty_text_input_field = empty_text_input.locator('input').first
    empty_text_input_field.fill('hello world')
    empty_text_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').nth(3)).to_have_text('value 4: hello world', use_inner_text=True)
    empty_text_input_field.focus()
    empty_text_input_field.clear()
    empty_text_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').nth(3)).to_have_text('value 4: ', use_inner_text=True)

def test_calls_callback_on_change(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that it correctly calls the callback on change.'
    text_input_field = app.get_by_test_id('stTextInput').nth(8).locator('input').first
    text_input_field.fill('hello world')
    text_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').nth(8)).to_have_text('value 9: hello world', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(9)).to_have_text('text input changed: True', use_inner_text=True)
    first_text_input_field = app.get_by_test_id('stTextInput').first.locator('input').first
    first_text_input_field.fill('hello world')
    first_text_input_field.press('Enter')
    expect(app.get_by_test_id('stMarkdown').first).to_have_text('value 1: hello world', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(8)).to_have_text('value 9: hello world', use_inner_text=True)
    expect(app.get_by_test_id('stMarkdown').nth(9)).to_have_text('text input changed: False', use_inner_text=True)