from typing import TYPE_CHECKING
from litestar import Litestar
if TYPE_CHECKING:
    from litestar.config.app import AppConfig

async def close_db_connection() -> None:
    """Closes the database connection on application shutdown."""

def receive_app_config(app_config: 'AppConfig') -> 'AppConfig':
    if False:
        for i in range(10):
            print('nop')
    'Receives parameters from the application.\n\n    In reality, this would be a library of boilerplate that is carried from one application to another, or a third-party\n    developed application configuration tool.\n    '
    app_config.on_shutdown.append(close_db_connection)
    return app_config
app = Litestar([], on_app_init=[receive_app_config])