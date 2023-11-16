import logging
from tests.conftest import TrackedContainer
LOGGER = logging.getLogger(__name__)

def check_r_mimetypes(container: TrackedContainer) -> None:
    if False:
        while True:
            i = 10
    'Check if Rscript command can be executed'
    LOGGER.info('Test that R command can be executed ...')
    Rcommand = 'if (length(getOption("jupyter.plot_mimetypes")) != 5) {stop("missing jupyter.plot_mimetypes")}'
    logs = container.run_and_wait(timeout=10, tty=True, command=['Rscript', '-e', Rcommand])
    LOGGER.debug(f'logs={logs!r}')
    assert len(logs) == 0, f'Command Rcommand={Rcommand!r} failed'