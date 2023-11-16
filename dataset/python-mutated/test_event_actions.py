"""Ensure stopPropagation and preventDefault work as expected."""
import asyncio
from typing import Callable, Coroutine, Generator
import pytest
from selenium.webdriver.common.by import By
from reflex.testing import AppHarness, WebDriver

def TestEventAction():
    if False:
        print('Hello World!')
    'App for testing event_actions.'
    import reflex as rx

    class EventActionState(rx.State):
        order: list[str]

        def on_click(self, ev):
            if False:
                return 10
            self.order.append(f'on_click:{ev}')

        def on_click2(self):
            if False:
                i = 10
                return i + 15
            self.order.append('on_click2')

    class EventFiringComponent(rx.Component):
        """A component that fires onClick event without passing DOM event."""
        tag = 'EventFiringComponent'

        def _get_custom_code(self) -> str | None:
            if False:
                while True:
                    i = 10
            return '\n                function EventFiringComponent(props) {\n                    return (\n                        <div id={props.id} onClick={(e) => props.onClick("foo")}>\n                            Event Firing Component\n                        </div>\n                    )\n                }'

        def get_event_triggers(self):
            if False:
                i = 10
                return i + 15
            return {'on_click': lambda : []}

    def index():
        if False:
            return 10
        return rx.vstack(rx.input(value=EventActionState.router.session.client_token, is_read_only=True, id='token'), rx.button('No events', id='btn-no-events'), rx.button('Stop Prop Only', id='btn-stop-prop-only', on_click=rx.stop_propagation), rx.button('Click event', on_click=EventActionState.on_click('no_event_actions'), id='btn-click-event'), rx.button('Click stop propagation', on_click=EventActionState.on_click('stop_propagation').stop_propagation, id='btn-click-stop-propagation'), rx.button('Click stop propagation2', on_click=EventActionState.on_click2.stop_propagation, id='btn-click-stop-propagation2'), rx.button('Click event 2', on_click=EventActionState.on_click2, id='btn-click-event2'), rx.link('Link', href='#', on_click=EventActionState.on_click('link_no_event_actions'), id='link'), rx.link('Link Stop Propagation', href='#', on_click=EventActionState.on_click('link_stop_propagation').stop_propagation, id='link-stop-propagation'), rx.link('Link Prevent Default Only', href='/invalid', on_click=rx.prevent_default, id='link-prevent-default-only'), rx.link('Link Prevent Default', href='/invalid', on_click=EventActionState.on_click('link_prevent_default').prevent_default, id='link-prevent-default'), rx.link('Link Both', href='/invalid', on_click=EventActionState.on_click('link_both').stop_propagation.prevent_default, id='link-stop-propagation-prevent-default'), EventFiringComponent.create(id='custom-stop-propagation', on_click=EventActionState.on_click('custom-stop-propagation').stop_propagation), EventFiringComponent.create(id='custom-prevent-default', on_click=EventActionState.on_click('custom-prevent-default').prevent_default), rx.list(rx.foreach(EventActionState.order, rx.list_item)), on_click=EventActionState.on_click('outer'))
    app = rx.App(state=EventActionState)
    app.add_page(index)
    app.compile()

@pytest.fixture(scope='session')
def event_action(tmp_path_factory) -> Generator[AppHarness, None, None]:
    if False:
        i = 10
        return i + 15
    'Start TestEventAction app at tmp_path via AppHarness.\n\n    Args:\n        tmp_path_factory: pytest tmp_path_factory fixture\n\n    Yields:\n        running AppHarness instance\n    '
    with AppHarness.create(root=tmp_path_factory.mktemp(f'event_action'), app_source=TestEventAction) as harness:
        yield harness

@pytest.fixture
def driver(event_action: AppHarness) -> Generator[WebDriver, None, None]:
    if False:
        print('Hello World!')
    'Get an instance of the browser open to the event_action app.\n\n    Args:\n        event_action: harness for TestEventAction app\n\n    Yields:\n        WebDriver instance.\n    '
    assert event_action.app_instance is not None, 'app is not running'
    driver = event_action.frontend()
    try:
        yield driver
    finally:
        driver.quit()

@pytest.fixture()
def token(event_action: AppHarness, driver: WebDriver) -> str:
    if False:
        return 10
    'Get the token associated with backend state.\n\n    Args:\n        event_action: harness for TestEventAction app.\n        driver: WebDriver instance.\n\n    Returns:\n        The token visible in the driver browser.\n    '
    assert event_action.app_instance is not None
    token_input = driver.find_element(By.ID, 'token')
    assert token_input
    token = event_action.poll_for_value(token_input)
    assert token is not None
    return token

@pytest.fixture()
def poll_for_order(event_action: AppHarness, token: str) -> Callable[[list[str]], Coroutine[None, None, None]]:
    if False:
        return 10
    'Poll for the order list to match the expected order.\n\n    Args:\n        event_action: harness for TestEventAction app.\n        token: The token visible in the driver browser.\n\n    Returns:\n        An async function that polls for the order list to match the expected order.\n    '

    async def _poll_for_order(exp_order: list[str]):

        async def _backend_state():
            return await event_action.get_state(token)

        async def _check():
            return (await _backend_state()).order == exp_order
        await AppHarness._poll_for_async(_check)
        assert (await _backend_state()).order == exp_order
    return _poll_for_order

@pytest.mark.parametrize(('element_id', 'exp_order'), [('btn-no-events', ['on_click:outer']), ('btn-stop-prop-only', []), ('btn-click-event', ['on_click:no_event_actions', 'on_click:outer']), ('btn-click-stop-propagation', ['on_click:stop_propagation']), ('btn-click-stop-propagation2', ['on_click2']), ('btn-click-event2', ['on_click2', 'on_click:outer']), ('link', ['on_click:link_no_event_actions', 'on_click:outer']), ('link-stop-propagation', ['on_click:link_stop_propagation']), ('link-prevent-default', ['on_click:link_prevent_default', 'on_click:outer']), ('link-prevent-default-only', ['on_click:outer']), ('link-stop-propagation-prevent-default', ['on_click:link_both']), ('custom-stop-propagation', ['on_click:custom-stop-propagation', 'on_click:outer']), ('custom-prevent-default', ['on_click:custom-prevent-default', 'on_click:outer'])])
@pytest.mark.usefixtures('token')
@pytest.mark.asyncio
async def test_event_actions(driver: WebDriver, poll_for_order: Callable[[list[str]], Coroutine[None, None, None]], element_id: str, exp_order: list[str]):
    """Click links and buttons and assert on fired events.

    Args:
        driver: WebDriver instance.
        poll_for_order: function that polls for the order list to match the expected order.
        element_id: The id of the element to click.
        exp_order: The expected order of events.
    """
    el = driver.find_element(By.ID, element_id)
    assert el
    prev_url = driver.current_url
    el.click()
    if 'on_click:outer' not in exp_order:
        await asyncio.sleep(0.5)
    await poll_for_order(exp_order)
    if element_id.startswith('link') and 'prevent-default' not in element_id:
        assert driver.current_url != prev_url
    else:
        assert driver.current_url == prev_url