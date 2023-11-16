from nicegui import ui

def main_demo() -> None:
    if False:
        return 10
    ui.mermaid('\n    graph LR;\n        A --> B;\n        A --> C;\n    ')