from nicegui import ui

def main_demo() -> None:
    if False:
        i = 10
        return i + 15
    ui.code("\n        from nicegui import ui\n        \n        ui.label('Code inception!')\n            \n        ui.run()\n    ").classes('w-full')