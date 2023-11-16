from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run

def get_first_graph_svg(app: Page):
    if False:
        while True:
            i = 10
    return app.locator('.stGraphVizChart > svg').nth(0)

def click_fullscreen(app: Page):
    if False:
        print('Hello World!')
    app.locator('[data-testid="StyledFullScreenButton"]').nth(0).click()
    app.wait_for_timeout(1000)

def test_initial_setup(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Initial setup: ensure charts are loaded.'
    wait_for_app_run(app)
    title_count = len(app.locator('.stGraphVizChart > svg > g > title').all())
    assert title_count == 6

def test_shows_left_and_right_graph(app: Page):
    if False:
        print('Hello World!')
    'Test if it shows left and right graph.'
    left_text = app.locator('.stGraphVizChart > svg > g > title').nth(3).text_content()
    right_text = app.locator('.stGraphVizChart > svg > g > title').nth(4).text_content()
    assert 'Left' in left_text and 'Right' in right_text

def test_first_graph_dimensions(app: Page):
    if False:
        i = 10
        return i + 15
    'Test the dimensions of the first graph.'
    first_graph_svg = get_first_graph_svg(app)
    assert first_graph_svg.get_attribute('width') == '79pt'
    assert first_graph_svg.get_attribute('height') == '116pt'

def test_first_graph_fullscreen(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test if the first graph shows in fullscreen.'
    app.locator('.stGraphVizChart').nth(0).hover()
    click_fullscreen(app)
    first_graph_svg = get_first_graph_svg(app)
    expect(first_graph_svg).not_to_have_attribute('width', '79pt')
    expect(first_graph_svg).not_to_have_attribute('height', '116pt')
    svg_dimensions = first_graph_svg.bounding_box()
    assert svg_dimensions['width'] == 1256
    assert svg_dimensions['height'] == 662
    assert_snapshot(first_graph_svg, name='graphviz_fullscreen')

def test_first_graph_after_exit_fullscreen(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        for i in range(10):
            print('nop')
    'Test if the first graph has correct size after exiting fullscreen.'
    app.locator('.stGraphVizChart').nth(0).hover()
    click_fullscreen(app)
    click_fullscreen(app)
    first_graph_svg = get_first_graph_svg(app)
    assert first_graph_svg.get_attribute('width') == '79pt'
    assert first_graph_svg.get_attribute('height') == '116pt'
    assert_snapshot(first_graph_svg, name='graphviz_after_exit_fullscreen')

def test_renders_with_specified_engines(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        print('Hello World!')
    'Test if it renders with specified engines.'
    engines = ['dot', 'neato', 'twopi', 'circo', 'fdp', 'osage', 'patchwork']
    radios = app.query_selector_all('label[data-baseweb="radio"]')
    for (idx, engine) in enumerate(engines):
        radios[idx].click(force=True)
        wait_for_app_run(app)
        expect(app.get_by_test_id('stMarkdown').nth(0)).to_have_text(engine)
        assert_snapshot(app.locator('.stGraphVizChart > svg').nth(2), name=f'st_graphviz_chart_engine-{engine}')

def test_dot_string(app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test if it renders charts when input is a string (dot language).'
    title = app.locator('.stGraphVizChart > svg > g > title').nth(5)
    expect(title).to_have_text('Dot')
    assert_snapshot(app.locator('.stGraphVizChart > svg').nth(5), name='st_graphviz_chart_dot_string')