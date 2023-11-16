# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from playwright.sync_api import Page, expect

from e2e_playwright.conftest import ImageCompareFunction, wait_for_app_run


def test_renders_chat_messages_correctly_1(
    themed_app: Page, assert_snapshot: ImageCompareFunction
):
    """Test if the chat messages render correctly"""
    # Wait a bit more to allow all images to load:
    chat_message_elements = themed_app.locator(".stChatMessage")
    expect(chat_message_elements).to_have_count(12)

    # rerun to populate session state chat message
    themed_app.keyboard.press("r")
    wait_for_app_run(themed_app, wait_delay=1000)
    expect(chat_message_elements).to_have_count(14)
    for i, element in enumerate(chat_message_elements.all()):
        element.scroll_into_view_if_needed()
        # Wait a bit more to allow the avatar images to load:
        themed_app.wait_for_timeout(100)
        assert_snapshot(element, name=f"chat_message-{i}")
