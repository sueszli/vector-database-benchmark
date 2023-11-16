from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_dataframe_input_format_rendering(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        i = 10
        return i + 15
    'Test that st.dataframe renders various data formats correctly via snapshot testing.'
    dataframe_elements = app.get_by_test_id('stDataFrame')
    expect(dataframe_elements).to_have_count(34)
    app.wait_for_timeout(250)
    for (i, element) in enumerate(dataframe_elements.all()):
        assert_snapshot(element, name=f'st_dataframe-input_data_{i}')