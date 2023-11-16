import logging
from tests.conftest import TrackedContainer
LOGGER = logging.getLogger(__name__)

def test_pandoc(container: TrackedContainer) -> None:
    if False:
        i = 10
        return i + 15
    'Pandoc shall be able to convert MD to HTML.'
    logs = container.run_and_wait(timeout=10, tty=True, command=['start.sh', 'bash', '-c', 'echo "**BOLD**" | pandoc'])
    assert '<p><strong>BOLD</strong></p>' in logs