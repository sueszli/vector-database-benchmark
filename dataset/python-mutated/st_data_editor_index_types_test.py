from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_data_editor_index_types(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test that st.data_editor renders various index types correctly.'
    dataframe_elements = app.get_by_test_id('stDataFrame')
    expect(dataframe_elements).to_have_count(7)
    app.wait_for_timeout(250)
    assert_snapshot(dataframe_elements.nth(0), name='st_data_editor-string_index')
    assert_snapshot(dataframe_elements.nth(1), name='st_data_editor-float64_index')
    assert_snapshot(dataframe_elements.nth(2), name='st_data_editor-int64_index')
    assert_snapshot(dataframe_elements.nth(3), name='st_data_editor-uint64_index')
    assert_snapshot(dataframe_elements.nth(4), name='st_data_editor-date_index')
    assert_snapshot(dataframe_elements.nth(5), name='st_data_editor-time_index')
    assert_snapshot(dataframe_elements.nth(6), name='st_data_editor-datetime_index')