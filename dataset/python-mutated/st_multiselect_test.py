from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run

def select_for_kth_multiselect(page: Page, option_text: str, k: int, close_after_selecting: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Select an option from a multiselect widget.\n\n    Parameters\n    ----------\n    page : Page\n        The playwright page to use.\n    option_text : str\n        The text of the option to select.\n    k : int\n        The index of the multiselect widget to select from.\n    close_after_selecting : bool\n        Whether to close the dropdown after selecting the option.\n    '
    multiselect_elem = page.locator('.stMultiSelect').nth(k)
    multiselect_elem.locator('input').click()
    page.locator('li').filter(has_text=option_text).first.click()
    if close_after_selecting:
        page.keyboard.press('Escape')
    wait_for_app_run(page)

def del_from_kth_multiselect(page: Page, option_text: str, k: int):
    if False:
        while True:
            i = 10
    'Delete an option from a multiselect widget.\n\n    Parameters\n    ----------\n    page : Page\n        The playwright page to use.\n    option_text : str\n        The text of the option to delete.\n    k : int\n        The index of the multiselect widget to delete from.\n    '
    multiselect_elem = page.locator('.stMultiSelect').nth(k)
    multiselect_elem.locator(f'span[data-baseweb="tag"] span[title="{option_text}"] + span[role="presentation"]').first.click()

def test_multiselect_on_load(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Should show widgets correctly when loaded.'
    multiselect_elements = themed_app.locator('.stMultiSelect')
    expect(multiselect_elements).to_have_count(12)
    for (idx, el) in enumerate(multiselect_elements.all()):
        assert_snapshot(el, name='multiselect-' + str(idx))

def test_multiselect_initial_value(app: Page):
    if False:
        while True:
            i = 10
    'Should show the correct initial values.'
    text_elements = app.locator("[data-testid='stText']")
    expect(text_elements).to_have_count(12)
    text_elements = text_elements.all_inner_texts()
    texts = [text.strip() for text in text_elements]
    expected = ['value 1: []', 'value 2: []', 'value 3: []', "value 4: ['tea', 'water']", 'value 5: []', 'value 6: []', 'value 7: []', 'value 8: []', 'value 9: []', 'value 10: []', 'value 11: []', 'multiselect changed: False']
    assert texts == expected

def test_multiselect_clear_all(app: Page):
    if False:
        print('Hello World!')
    'Should clear all options when clicking clear all.'
    select_for_kth_multiselect(app, 'Female', 1, True)
    app.locator('.stMultiSelect [role="button"][aria-label="Clear all"]').first.click()
    expect(app.locator("[data-testid='stText']").nth(1)).to_have_text('value 2: []')

def test_multiselect_show_values_in_dropdown(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Screenshot test to check that values are shown in dropdown.'
    multiselect_elem = app.locator('.stMultiSelect').nth(0)
    multiselect_elem.locator('input').click()
    dropdown_elems = app.locator('li').all()
    assert len(dropdown_elems) == 2
    for (idx, el) in enumerate(dropdown_elems):
        assert_snapshot(el, name='multiselect-dropdown-' + str(idx))

def test_multiselect_long_values_in_dropdown(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Should show long values correctly (with ellipses) in the dropdown menu.'
    multiselect_elem = app.locator('.stMultiSelect').nth(4)
    multiselect_elem.locator('input').click()
    dropdown_elems = app.locator('li').all()
    for (idx, el) in enumerate(dropdown_elems):
        assert_snapshot(el, name='multiselect-dropdown-long-label-' + str(idx))

def test_multiselect_register_callback(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Should call the callback when an option is selected.'
    app.locator('.stMultiSelect').nth(10).locator('input').click()
    app.locator('li').first.click()
    expect(app.locator("[data-testid='stText']").nth(10)).to_have_text("value 11: ['male']")
    expect(app.locator("[data-testid='stText']").nth(11)).to_have_text('multiselect changed: True')

def test_multiselect_max_selections_form(app: Page):
    if False:
        while True:
            i = 10
    'Should apply max selections when used in form.'
    select_for_kth_multiselect(app, 'male', 8, False)
    expect(app.locator('li')).to_have_text('You can only select up to 1 option. Remove an option first.', use_inner_text=True)

def test_multiselect_max_selections_1(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Should show the correct text when maxSelections is reached and closing after selecting.'
    select_for_kth_multiselect(app, 'male', 9, True)
    app.locator('.stMultiSelect').nth(9).click()
    expect(app.locator('li')).to_have_text('You can only select up to 1 option. Remove an option first.', use_inner_text=True)

def test_multiselect_max_selections_2(app: Page):
    if False:
        while True:
            i = 10
    'Should show the correct text when maxSelections is reached and not closing after selecting.'
    select_for_kth_multiselect(app, 'male', 9, False)
    expect(app.locator('li')).to_have_text('You can only select up to 1 option. Remove an option first.', use_inner_text=True)

def test_multiselect_valid_options(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Should allow selections when there are valid options.'
    expect(app.locator('.stMultiSelect').first).to_have_text('multiselect 1\n\nPlease select', use_inner_text=True)

def test_multiselect_no_valid_options(app: Page):
    if False:
        return 10
    'Should show that their are no options.'
    expect(app.locator('.stMultiSelect').nth(2)).to_have_text('multiselect 3\n\nNo options to select.', use_inner_text=True)

def test_multiselect_single_selection(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Should allow selections.'
    select_for_kth_multiselect(app, 'Female', 1, True)
    expect(app.locator('.stMultiSelect span').nth(1)).to_have_text('Female', use_inner_text=True)
    assert_snapshot(app.locator('.stMultiSelect').nth(1), name='multiselect-selection')
    expect(app.locator("[data-testid='stText']").nth(1)).to_have_text("value 2: ['female']", use_inner_text=True)

def test_multiselect_deselect_option(app: Page):
    if False:
        while True:
            i = 10
    'Should deselect an option when deselecting it.'
    select_for_kth_multiselect(app, 'Female', 1, True)
    select_for_kth_multiselect(app, 'Male', 1, True)
    del_from_kth_multiselect(app, 'Female', 1)
    expect(app.locator("[data-testid='stText']").nth(1)).to_have_text("value 2: ['male']")

def test_multiselect_option_over_max_selections(app: Page):
    if False:
        i = 10
        return i + 15
    'Should show an error when more than max_selections got selected.'
    app.locator('.stCheckbox').first.click()
    expect(app.locator('.element-container .stException')).to_contain_text('Multiselect has 2 options selected but max_selections\nis set to 1')

def test_multiselect_double_selection(app: Page):
    if False:
        return 10
    'Should allow multiple selections.'
    select_for_kth_multiselect(app, 'Female', 1, True)
    select_for_kth_multiselect(app, 'Male', 1, True)
    expect(app.locator("[data-testid='stText']").nth(1)).to_have_text("value 2: ['female', 'male']")