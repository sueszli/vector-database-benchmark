from docs.examples.application_hooks.on_app_init import app, close_db_connection

def test_on_app_init() -> None:
    if False:
        while True:
            i = 10
    assert close_db_connection in app.on_shutdown