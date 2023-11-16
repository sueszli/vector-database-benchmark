import os
import pytest
from wiki import main
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

@pytest.mark.flaky
def test_main():
    if False:
        print('Hello World!')
    main(PROJECT)