from playwright.sync_api import Page, expect
from e2e_playwright.conftest import wait_for_app_run

def test_regression_with_file_uploader_and_chat_input(app: Page):
    if False:
        for i in range(10):
            print('nop')
    'Test issue described in https://github.com/streamlit/streamlit/issues/7556.'
    chat_input_element = app.get_by_test_id('stChatInput').first
    chat_input_element.fill('Hello world!')
    chat_input_element.press('Enter')
    wait_for_app_run(app)
    last_chat_message = app.get_by_test_id('stChatMessageContent').last
    expect(last_chat_message).to_have_text('Good at 1', use_inner_text=True)