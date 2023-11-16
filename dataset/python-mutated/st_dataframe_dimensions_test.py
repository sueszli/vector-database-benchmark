from playwright.sync_api import Page, expect

def test_data_frame_with_different_sizes(app: Page):
    if False:
        i = 10
        return i + 15
    'Test that st.dataframe should show different sizes as expected.'
    expected = [{'width': '704px', 'height': '400px'}, {'width': '250px', 'height': '150px'}, {'width': '250px', 'height': '400px'}, {'width': '704px', 'height': '150px'}, {'width': '704px', 'height': '5000px'}, {'width': '704px', 'height': '400px'}, {'width': '500px', 'height': '400px'}, {'width': '704px', 'height': '400px'}, {'width': '704px', 'height': '400px'}, {'width': '200px', 'height': '400px'}, {'width': '704px', 'height': '400px'}, {'width': '704px', 'height': '400px'}]
    dataframe_elements = app.locator('.stDataFrame')
    expect(dataframe_elements).to_have_count(12)
    for (i, element) in enumerate(dataframe_elements.all()):
        expect(element).to_have_css('width', expected[i]['width'])
        expect(element).to_have_css('height', expected[i]['height'])