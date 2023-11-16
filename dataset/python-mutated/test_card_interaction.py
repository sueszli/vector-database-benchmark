import os
import signal
from utils import start_waved, AppRunner
import pytest
from playwright.sync_api import Page, expect

@pytest.fixture(scope='session', autouse=True)
def global_setup_teardown(playwright):
    if False:
        i = 10
        return i + 15
    playwright.selectors.set_test_id_attribute('data-test')
    expect.set_options(timeout=10000)

@pytest.fixture(scope='module', autouse=True)
def setup_teardown():
    if False:
        i = 10
        return i + 15
    waved_p = None
    try:
        waved_p = start_waved()
        yield
    finally:
        if waved_p:
            os.killpg(os.getpgid(waved_p.pid), signal.SIGTERM)

def test_interactions(page: Page):
    if False:
        for i in range(10):
            print('nop')
    code = '\nfrom h2o_wave import Q, ui, main, app\n\n\n@app(\'/\')\nasync def serve(q: Q):\n    if not q.client.initialized:  # First visit, create an empty form card for our wizard\n        q.page[\'wizard\'] = ui.form_card(box=\'1 1 2 4\', items=[])\n        q.client.initialized = True\n\n    wizard = q.page[\'wizard\']  # Get a reference to the wizard form\n    if q.args.step1:\n        wizard.items = [\n            ui.text_xl(\'Wizard - Step 1\'),\n            ui.text(\'What is your name?\', name=\'text\'),\n            ui.textbox(name=\'nickname\', label=\'My name is...\', value=\'Gandalf\'),\n            ui.buttons([ui.button(name=\'step2\', label=\'Next\', primary=True)]),\n        ]\n    elif q.args.step2:\n        q.client.nickname = q.args.nickname\n        wizard.items = [\n            ui.text_xl(\'Wizard - Step 2\'),\n            ui.text(f\'Hi {q.args.nickname}! How do you feel right now?\', name=\'text\'),\n            ui.textbox(name=\'feeling\', label=\'I feel...\', value=\'magical\'),\n            ui.buttons([ui.button(name=\'step3\', label=\'Next\', primary=True)]),\n        ]\n    elif q.args.step3:\n        wizard.items = [\n            ui.text_xl(\'Wizard - Done\'),\n            ui.text(\n                f\'What a coincidence, {q.client.nickname}! I feel {q.args.feeling} too!\',\n                name=\'text\',\n            ),\n            ui.buttons([ui.button(name=\'step1\', label=\'Try Again\', primary=True)]),\n        ]\n    else:\n        wizard.items = [\n            ui.text_xl(\'Wizard Example\'),\n            ui.text("Let\'s have a conversation, shall we?"),\n            ui.buttons([ui.button(name=\'step1\', label=\'Of course!\', primary=True)]),\n        ]\n\n    await q.page.save()\n'
    with AppRunner(code):
        page.goto('http://localhost:10101')
        expect(page.get_by_text('Wizard Example')).to_be_visible()
        page.get_by_text('Of course!').click()
        expect(page.get_by_text('What is your name?')).to_be_visible()
        page.get_by_test_id('nickname').fill('Fred')
        page.locator('text=Next').click()
        expect(page.locator('text=Hi Fred! How do you feel right now?')).to_be_visible()
        page.get_by_test_id('feeling').fill('happy')
        page.locator('text=Next').click()
        expect(page.locator('text=What a coincidence, Fred! I feel happy too!')).to_be_visible()