from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_code_display(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that st.code displays a code block.'
    code_element = app.locator('.element-container pre').first
    expect(code_element).to_contain_text('This code is awesome!')

def test_syntax_highlighting(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that the copy-to-clipboard action appears on hover.'
    first_code_element = themed_app.locator('.element-container:first-child pre').first
    first_code_element.hover()
    assert_snapshot(first_code_element, name='syntax_highlighting-hover')

def test_code_blocks_render_correctly(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test that the code blocks render as expected via screenshot matching.'
    code_blocks = themed_app.get_by_test_id('stCodeBlock')
    assert_snapshot(code_blocks.nth(0), name='st_code-auto_lang')
    assert_snapshot(code_blocks.nth(1), name='st_code-empty')
    assert_snapshot(code_blocks.nth(2), name='st_code-python_lang')
    assert_snapshot(code_blocks.nth(3), name='st_code-line_numbers')
    assert_snapshot(code_blocks.nth(4), name='st_code-no_lang')
    assert_snapshot(code_blocks.nth(5), name='st_markdown-code_block')

def test_correct_bottom_spacing_for_code_blocks(app: Page):
    if False:
        return 10
    'Test that the code blocks have the correct bottom spacing.'
    expect(app.get_by_test_id('stExpander').nth(0).get_by_test_id('stCodeBlock').first).to_have_css('margin-bottom', '0px')
    expect(app.get_by_test_id('stExpander').nth(1).get_by_test_id('stCodeBlock').first).to_have_css('margin-bottom', '16px')