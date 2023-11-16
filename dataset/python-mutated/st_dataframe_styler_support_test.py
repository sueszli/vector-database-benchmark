from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_dataframe_pd_styler(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test that st.dataframe supports styling and display values via Pandas Styler.'
    elements = themed_app.get_by_test_id('stDataFrame')
    expect(elements).to_have_count(4)
    themed_app.wait_for_timeout(250)
    assert_snapshot(elements.nth(0), name='st_dataframe-styler_value_formatting')
    assert_snapshot(elements.nth(1), name='st_dataframe-styler_background_color')
    assert_snapshot(elements.nth(2), name='st_dataframe-styler_background_and_font')
    assert_snapshot(elements.nth(3), name='st_dataframe-styler_gradient')