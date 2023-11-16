import os
import pytest
from blog import main
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.mark.flaky
def test_main():
    if False:
        print('Hello World!')
    main(PROJECT)