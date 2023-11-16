from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction

def test_data_editor_input_format_rendering(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test that st.data_editor renders various data formats correctly via snapshot testing.'
    dataframe_elements = app.get_by_test_id('stDataFrame')
    expect(dataframe_elements).to_have_count(35)
    app.wait_for_timeout(1000)
    for (i, element) in enumerate(dataframe_elements.all()):
        assert_snapshot(element, name=f'st_data_editor-input_data_{i}')