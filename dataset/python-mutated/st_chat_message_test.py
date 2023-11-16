from playwright.sync_api import Page, expect
from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run

def test_renders_chat_messages_correctly_1(themed_app: Page, assert_snapshot: ImageCompareFunction):
    if False:
        return 10
    'Test if the chat messages render correctly'
    chat_message_elements = themed_app.locator('.stChatMessage')
    expect(chat_message_elements).to_have_count(12)
    themed_app.keyboard.press('r')
    wait_for_app_run(themed_app, wait_delay=1000)
    expect(chat_message_elements).to_have_count(14)
    for (i, element) in enumerate(chat_message_elements.all()):
        element.scroll_into_view_if_needed()
        themed_app.wait_for_timeout(100)
        assert_snapshot(element, name=f'chat_message-{i}')