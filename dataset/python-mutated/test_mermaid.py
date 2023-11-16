from nicegui import ui
from .screen import Screen

def test_mermaid(screen: Screen):
    if False:
        while True:
            i = 10
    m = ui.mermaid('\n        graph TD;\n            Node_A --> Node_B;\n    ')
    screen.open('/')
    assert screen.find('Node_A').get_attribute('class') == 'nodeLabel'
    m.set_content('\ngraph TD;\n    Node_C --> Node_D;\n')
    assert screen.find('Node_C').get_attribute('class') == 'nodeLabel'
    screen.should_not_contain('Node_A')

def test_mermaid_with_line_breaks(screen: Screen):
    if False:
        i = 10
        return i + 15
    ui.mermaid('\n        requirementDiagram\n\n        requirement test_req {\n            id: 1\n            text: some test text\n            risk: high\n            verifymethod: test\n        }\n    ')
    screen.open('/')
    screen.should_contain('<<Requirement>>')
    screen.should_contain('Id: 1')
    screen.should_contain('Text: some test text')
    screen.should_contain('Risk: High')
    screen.should_contain('Verification: Test')

def test_replace_mermaid(screen: Screen):
    if False:
        i = 10
        return i + 15
    with ui.row() as container:
        ui.mermaid('graph LR; Node_A')

    def replace():
        if False:
            return 10
        container.clear()
        with container:
            ui.mermaid('graph LR; Node_B')
    ui.button('Replace', on_click=replace)
    screen.open('/')
    screen.should_contain('Node_A')
    screen.click('Replace')
    screen.wait(0.5)
    screen.should_contain('Node_B')
    screen.should_not_contain('Node_A')

def test_create_dynamically(screen: Screen):
    if False:
        return 10
    ui.button('Create', on_click=lambda : ui.mermaid('graph LR; Node'))
    screen.open('/')
    screen.click('Create')
    screen.should_contain('Node')