import pytest
from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_dataframe_toolbar_on_hover(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Test that the toolbar is shown when hovering over a dataframe.'
    dataframe_element = themed_app.get_by_test_id('stDataFrame').nth(0)
    dataframe_toolbar = dataframe_element.get_by_test_id('stElementToolbar')
    expect(dataframe_toolbar).to_have_css('opacity', '0')
    dataframe_element.hover()
    expect(dataframe_toolbar).to_have_css('opacity', '1')
    assert_snapshot(dataframe_toolbar, name='st_dataframe-toolbar')

def test_data_editor_toolbar_on_hover(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that the toolbar is shown when hovering over a data editor component.'
    data_editor_element = themed_app.get_by_test_id('stDataFrame').nth(1)
    data_editor_toolbar = data_editor_element.get_by_test_id('stElementToolbar')
    expect(data_editor_toolbar).to_have_css('opacity', '0')
    data_editor_element.hover()
    expect(data_editor_toolbar).to_have_css('opacity', '1')
    assert_snapshot(data_editor_toolbar, name='st_data_editor-toolbar')

def test_data_editor_delete_row_via_toolbar(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test that a row can be deleted via the toolbar.'
    data_editor_element = themed_app.get_by_test_id('stDataFrame').nth(1)
    data_editor_toolbar = data_editor_element.get_by_test_id('stElementToolbar')
    data_editor_element.click(position={'x': 10, 'y': 100})
    assert_snapshot(data_editor_element, name='st_data_editor-selected_row_for_deletion')
    expect(data_editor_element).to_have_css('height', '248px')
    expect(data_editor_toolbar).to_have_css('opacity', '1')
    assert_snapshot(data_editor_toolbar, name='st_data_editor-row_deletion_toolbar')
    delete_row_button = data_editor_toolbar.get_by_test_id('stElementToolbarButton').nth(0)
    delete_row_button.click()
    expect(data_editor_element).to_have_css('height', '213px')

def test_data_editor_delete_row_via_hotkey(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that a row can be deleted via delete hotkey.'
    data_editor_element = app.get_by_test_id('stDataFrame').nth(1)
    expect(data_editor_element).to_have_css('height', '248px')
    data_editor_element.click(position={'x': 10, 'y': 100})
    data_editor_element.press('Delete')
    expect(data_editor_element).to_have_css('height', '213px')

def test_data_editor_add_row_via_toolbar(app: Page):
    if False:
        while True:
            i = 10
    'Test that a row can be added via the toolbar.'
    data_editor_element = app.get_by_test_id('stDataFrame').nth(1)
    data_editor_toolbar = data_editor_element.get_by_test_id('stElementToolbar')
    expect(data_editor_element).to_have_css('height', '248px')
    data_editor_element.hover()
    expect(data_editor_toolbar).to_have_css('opacity', '1')
    add_row_button = data_editor_toolbar.get_by_test_id('stElementToolbarButton').nth(0)
    add_row_button.click()
    expect(data_editor_element).to_have_css('height', '283px')

def test_data_editor_add_row_via_trailing_row(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that a row can be added by clicking on the trailing row.'
    data_editor_element = app.get_by_test_id('stDataFrame').nth(1)
    expect(data_editor_element).to_have_css('height', '248px')
    data_editor_element.click(position={'x': 40, 'y': 220})
    expect(data_editor_element).to_have_css('height', '283px')

@pytest.mark.skip_browser('firefox')
def test_dataframe_toolbar_on_toolbar_hover(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that the toolbar is shown when hovering over the toolbar.'
    dataframe_element = app.get_by_test_id('stDataFrame').nth(0)
    dataframe_toolbar = dataframe_element.get_by_test_id('stElementToolbar')
    expect(dataframe_toolbar).to_have_css('opacity', '0')
    dataframe_toolbar.hover(force=True, position={'x': 0, 'y': 0})
    expect(dataframe_toolbar).to_have_css('opacity', '1')

def test_open_search_via_toolbar(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Test that clicking on search toolbar button triggers dataframe search.'
    dataframe_element = themed_app.get_by_test_id('stDataFrame').nth(0)
    dataframe_toolbar = dataframe_element.get_by_test_id('stElementToolbar')
    search_toolbar_button = dataframe_toolbar.get_by_test_id('stElementToolbarButton').nth(1)
    dataframe_element.hover()
    expect(dataframe_toolbar).to_have_css('opacity', '1')
    search_toolbar_button.hover()
    expect(themed_app.get_by_test_id('stTooltipContent')).to_have_text('Search')
    assert_snapshot(dataframe_toolbar, name='st_dataframe-toolbar_hover_search')
    search_toolbar_button.click()
    assert_snapshot(dataframe_element, name='st_dataframe-trigger_search_via_toolbar')

def test_open_search_via_hotkey(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that the search can be opened via a hotkey.'
    dataframe_element = app.get_by_test_id('stDataFrame').nth(0)
    dataframe_element.press('Control+F')
    assert_snapshot(dataframe_element, name='st_dataframe-trigger_search_via_hotkey')

def test_clicking_on_fullscreen_toolbar_button(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that clicking on fullscreen toolbar button expands the dataframe into fullscreen.'
    dataframe_element = app.get_by_test_id('stDataFrame').nth(0)
    dataframe_toolbar = dataframe_element.get_by_test_id('stElementToolbar')
    fullscreen_wrapper = app.get_by_test_id('stStyledFullScreenFrame').nth(0)
    fullscreen_toolbar_button = dataframe_toolbar.get_by_test_id('stElementToolbarButton').nth(2)
    dataframe_element.hover()
    expect(dataframe_toolbar).to_have_css('opacity', '1')
    fullscreen_toolbar_button.click()
    assert_snapshot(fullscreen_wrapper, name='st_dataframe-fullscreen_expanded')
    fullscreen_toolbar_button.click()
    assert_snapshot(fullscreen_wrapper, name='st_dataframe-fullscreen_collapsed')