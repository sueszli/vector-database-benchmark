import os
import signal
from utils import start_waved, AppRunner
import pytest
from playwright.sync_api import Page, expect

@pytest.fixture(scope='module', autouse=True)
def setup_teardown():
    if False:
        print('Hello World!')
    waved_p = None
    expect.set_options(timeout=10000)
    try:
        waved_p = start_waved()
        yield
    finally:
        if waved_p:
            os.killpg(os.getpgid(waved_p.pid), signal.SIGTERM)

def test_by_name_updates(page: Page):
    if False:
        print('Hello World!')
    code = "\nfrom h2o_wave import Q, ui, main, app\n\n\n@app('/')\nasync def serve(q: Q):\n    q.page['wizard'] = ui.form_card(box='1 1 2 4', items=[\n        ui.text_xl(name='text_name', content='Wizard'),\n        ui.inline(items=[\n            ui.button(name='back', label='Back'),\n        ]),\n    ])\n    q.page['wizard'].text_name.content = 'foo1'\n    q.page['wizard'].back.label = 'foo2'\n\n    q.page['header'] = ui.header_card(box='4 6 4 1', title='Header', subtitle='Subtitle', secondary_items=[\n        ui.button(name='button_name', label='Button'),\n    ])\n    q.page['header'].button_name.label = 'foo3'\n\n    q.page['example'] = ui.form_card(box='5 1 4 5', items=[\n        ui.buttons([\n            ui.button(name='primary_button', label='Primary', primary=True),\n        ]),\n    ])\n    q.page['example'].primary_button.label = 'foo4'\n\n    q.page['nav'] = ui.tab_card(\n        box='1 6 4 1',\n        items=[\n            ui.tab(name='#hash', label='Spam'),\n            ui.tab(name='plaintext', label='Ham'),\n        ],\n    )\n    q.page['nav']['#hash'].label = 'foo5'\n    q.page['nav'].plaintext.label = 'foo6'\n\n    await q.page.save()\n\n"
    with AppRunner(code):
        page.goto('http://localhost:10101')
        expect(page.get_by_text('foo1')).to_be_visible()
        expect(page.get_by_text('foo2')).to_be_visible()
        expect(page.get_by_text('foo3')).to_be_visible()
        expect(page.get_by_text('foo4')).to_be_visible()
        expect(page.get_by_text('foo5')).to_be_visible()
        expect(page.get_by_text('foo6')).to_be_visible()

def test_by_name_updates_dialog_init(page: Page):
    if False:
        print('Hello World!')
    code = "\nfrom h2o_wave import main, app, Q, ui\n\n\n@app('/')\nasync def serve(q: Q):\n    q.page['meta'] = ui.meta_card(box='', dialog=ui.dialog(title='Order Donuts', items=[\n        ui.button(name='next_step', label='Next')\n    ]))\n    q.page['meta'].next_step.label = 'New next'\n\n    await q.page.save()\n"
    with AppRunner(code):
        page.goto('http://localhost:10101')
        expect(page.get_by_text('New next')).to_be_visible()

def test_by_name_updates_dialog(page: Page):
    if False:
        for i in range(10):
            print('nop')
    code = "\nfrom h2o_wave import main, app, Q, ui\n\n\n@app('/')\nasync def serve(q: Q):\n    q.page['meta'] = ui.meta_card(box='')\n    q.page['meta'].dialog = ui.dialog(title='Order Donuts', items=[\n        ui.button(name='next_step', label='Next')\n    ])\n    q.page['meta'].next_step.label = 'New next'\n\n    await q.page.save()\n"
    with AppRunner(code):
        page.goto('http://localhost:10101')
        expect(page.get_by_text('New next')).to_be_visible()

def test_by_name_updates_side_panel_init(page: Page):
    if False:
        print('Hello World!')
    code = "\nfrom h2o_wave import main, app, Q, ui\n\n\n@app('/')\nasync def serve(q: Q):\n    q.page['meta'] = ui.meta_card(box='', side_panel=ui.side_panel(title='Order Donuts', items=[\n        ui.button(name='next_step', label='Next')\n    ]))\n    q.page['meta'].next_step.label = 'New next'\n\n    await q.page.save()\n"
    with AppRunner(code):
        page.goto('http://localhost:10101')
        expect(page.get_by_text('New next')).to_be_visible()

def test_by_name_updates_side_panel(page: Page):
    if False:
        i = 10
        return i + 15
    code = "\nfrom h2o_wave import main, app, Q, ui\n\n\n@app('/')\nasync def serve(q: Q):\n    q.page['meta'] = ui.meta_card(box='')\n    q.page['meta'].side_panel = ui.side_panel(title='Order Donuts', items=[\n        ui.button(name='next_step', label='Next')\n    ])\n    q.page['meta'].next_step.label = 'New next'\n\n    await q.page.save()\n"
    with AppRunner(code):
        page.goto('http://localhost:10101')
        expect(page.get_by_text('New next')).to_be_visible()

def test_by_name_updates_notification_bar_init(page: Page):
    if False:
        while True:
            i = 10
    code = "\nfrom h2o_wave import main, app, Q, ui\n\n\n@app('/')\nasync def serve(q: Q):\n    q.page['meta'] = ui.meta_card(box='', notification_bar=ui.notification_bar(\n        text='Success notification',\n        buttons=[ui.button(name='btn1', label='Button 1')]\n    ))\n    q.page['meta'].btn1.label = 'New text'\n\n    await q.page.save()\n"
    with AppRunner(code):
        page.goto('http://localhost:10101')
        expect(page.get_by_text('New text')).to_be_visible()

def test_by_name_updates_notification_bar(page: Page):
    if False:
        while True:
            i = 10
    code = "\nfrom h2o_wave import main, app, Q, ui\n\n\n@app('/')\nasync def serve(q: Q):\n    q.page['meta'] = ui.meta_card(box='')\n    q.page['meta'].notification_bar = ui.notification_bar(\n        text='Success notification',\n        buttons=[ui.button(name='btn1', label='Button 1')]\n    )\n    q.page['meta'].btn1.label = 'New text'\n\n    await q.page.save()\n"
    with AppRunner(code):
        page.goto('http://localhost:10101')
        expect(page.get_by_text('New text')).to_be_visible()

def test_by_name_updates_card_commands(page: Page):
    if False:
        i = 10
        return i + 15
    code = "\nfrom h2o_wave import main, app, Q, ui\n\n\n@app('/')\nasync def serve(q: Q):\n    q.page['form'] = ui.form_card(\n        box='1 1 3 3',\n        items=[],\n        commands=[\n            ui.command(name='step1', label='Step 1'),\n        ]\n    )\n    q.page['form'].step1.label = 'New text'\n    await q.page.save()\n"
    with AppRunner(code):
        page.goto('http://localhost:10101')
        page.click('[data-test="form"]:nth-child(2) > div')
        expect(page.get_by_text('New text')).to_be_visible()