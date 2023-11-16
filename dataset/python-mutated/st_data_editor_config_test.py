from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_data_editor_supports_various_configurations(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Screenshot test that st.data_editor supports various configuration options.'
    elements = app.get_by_test_id('stDataFrame')
    expect(elements).to_have_count(22)
    app.wait_for_timeout(500)
    assert_snapshot(elements.nth(0), name='st_data_editor-disabled_all_columns')
    assert_snapshot(elements.nth(1), name='st_data_editor-disabled_two_columns')
    assert_snapshot(elements.nth(2), name='st_data_editor-hide_index')
    assert_snapshot(elements.nth(3), name='st_data_editor-show_index')
    assert_snapshot(elements.nth(4), name='st_data_editor-custom_column_order')
    assert_snapshot(elements.nth(5), name='st_data_editor-column_labels')
    assert_snapshot(elements.nth(6), name='st_data_editor-hide_columns')
    assert_snapshot(elements.nth(7), name='st_data_editor-set_column_width')
    assert_snapshot(elements.nth(8), name='st_data_editor-help_tooltips')
    assert_snapshot(elements.nth(9), name='st_data_editor-text_column')
    assert_snapshot(elements.nth(10), name='st_data_editor-number_column')
    assert_snapshot(elements.nth(11), name='st_data_editor-checkbox_column')
    assert_snapshot(elements.nth(12), name='st_data_editor-selectbox_column')
    assert_snapshot(elements.nth(13), name='st_data_editor-link_column')
    assert_snapshot(elements.nth(14), name='st_data_editor-datetime_column')
    assert_snapshot(elements.nth(15), name='st_data_editor-date_column')
    assert_snapshot(elements.nth(16), name='st_data_editor-time_column')
    assert_snapshot(elements.nth(17), name='st_data_editor-progress_column')
    assert_snapshot(elements.nth(18), name='st_data_editor-list_column')
    assert_snapshot(elements.nth(19), name='st_data_editor-bar_chart_column')
    assert_snapshot(elements.nth(20), name='st_data_editor-line_chart_column')
    assert_snapshot(elements.nth(21), name='st_data_editor-image_column')