from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_dataframe_index_types(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test that st.dataframe render various index types correctly.'
    dataframe_elements = app.get_by_test_id('stDataFrame')
    expect(dataframe_elements).to_have_count(13)
    app.wait_for_timeout(250)
    assert_snapshot(dataframe_elements.nth(0), name='st_dataframe-string_index')
    assert_snapshot(dataframe_elements.nth(1), name='st_dataframe-float64_index')
    assert_snapshot(dataframe_elements.nth(2), name='st_dataframe-int64_index')
    assert_snapshot(dataframe_elements.nth(3), name='st_dataframe-uint64_index')
    assert_snapshot(dataframe_elements.nth(4), name='st_dataframe-datetime_index')
    assert_snapshot(dataframe_elements.nth(5), name='st_dataframe-date_index')
    assert_snapshot(dataframe_elements.nth(6), name='st_dataframe-time_index')
    assert_snapshot(dataframe_elements.nth(7), name='st_dataframe-interval_index')
    assert_snapshot(dataframe_elements.nth(8), name='st_dataframe-list_index')
    assert_snapshot(dataframe_elements.nth(9), name='st_dataframe-multi_index')
    assert_snapshot(dataframe_elements.nth(10), name='st_dataframe-categorical')
    assert_snapshot(dataframe_elements.nth(11), name='st_dataframe-period_index')
    assert_snapshot(dataframe_elements.nth(12), name='st_dataframe-timedelta_index')