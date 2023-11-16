from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_dataframe_supports_various_configurations(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Screenshot test that st.dataframe supports various configuration options.'
    dataframe_elements = themed_app.get_by_test_id('stDataFrame')
    expect(dataframe_elements).to_have_count(22)
    themed_app.wait_for_timeout(250)
    assert_snapshot(dataframe_elements.nth(0), name='st_dataframe-hide_index')
    assert_snapshot(dataframe_elements.nth(1), name='st_dataframe-show_index')
    assert_snapshot(dataframe_elements.nth(2), name='st_dataframe-custom_column_order')
    assert_snapshot(dataframe_elements.nth(3), name='st_dataframe-column_labels')
    assert_snapshot(dataframe_elements.nth(4), name='st_dataframe-hide_columns')
    assert_snapshot(dataframe_elements.nth(5), name='st_dataframe-set_column_width')
    assert_snapshot(dataframe_elements.nth(6), name='st_dataframe-help_tooltips')
    assert_snapshot(dataframe_elements.nth(7), name='st_dataframe-ignore_edit_options')
    assert_snapshot(dataframe_elements.nth(8), name='st_dataframe-text_column')
    assert_snapshot(dataframe_elements.nth(9), name='st_dataframe-number_column')
    assert_snapshot(dataframe_elements.nth(10), name='st_dataframe-checkbox_column')
    assert_snapshot(dataframe_elements.nth(11), name='st_dataframe-selectbox_column')
    assert_snapshot(dataframe_elements.nth(12), name='st_dataframe-link_column')
    assert_snapshot(dataframe_elements.nth(13), name='st_dataframe-datetime_column')
    assert_snapshot(dataframe_elements.nth(14), name='st_dataframe-date_column')
    assert_snapshot(dataframe_elements.nth(15), name='st_dataframe-time_column')
    assert_snapshot(dataframe_elements.nth(16), name='st_dataframe-progress_column')
    assert_snapshot(dataframe_elements.nth(17), name='st_dataframe-list_column')
    assert_snapshot(dataframe_elements.nth(18), name='st_dataframe-bar_chart_column')
    assert_snapshot(dataframe_elements.nth(19), name='st_dataframe-line_chart_column')
    assert_snapshot(dataframe_elements.nth(20), name='st_dataframe-image_column')
    assert_snapshot(dataframe_elements.nth(21), name='st_dataframe-auto_sized_columns')