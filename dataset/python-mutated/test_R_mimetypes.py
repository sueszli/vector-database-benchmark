from tests.conftest import TrackedContainer
from tests.R_mimetype_check import check_r_mimetypes

def test_mimetypes(container: TrackedContainer) -> None:
    if False:
        return 10
    'Check if Rscript command for mimetypes can be executed'
    check_r_mimetypes(container)