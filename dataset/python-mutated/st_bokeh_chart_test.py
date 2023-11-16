from playwright.sync_api import Page, expect

def test_bokeh_chart(themed_app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test that st.bokeh_chart renders correctly.'
    bokeh_chart_elements = themed_app.locator('[data-testid=stBokehChart]')
    expect(bokeh_chart_elements).to_have_count(4)
    expect(bokeh_chart_elements.nth(0).locator('canvas').nth(0)).to_be_visible()
    expect(bokeh_chart_elements.nth(1).locator('canvas').nth(0)).to_be_visible()
    expect(bokeh_chart_elements.nth(2).locator('canvas').nth(0)).to_be_visible()
    expect(bokeh_chart_elements.nth(3).locator('canvas').nth(0)).to_be_visible()