from nicegui import ui

def main_demo() -> None:
    if False:
        while True:
            i = 10
    from nicegui import app
    app.add_static_files('/examples', 'examples')
    ui.label('Some NiceGUI Examples').classes('text-h5')
    ui.link('AI interface', '/examples/ai_interface/main.py')
    ui.link('Custom FastAPI app', '/examples/fastapi/main.py')
    ui.link('Authentication', '/examples/authentication/main.py')