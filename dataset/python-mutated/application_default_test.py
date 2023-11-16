import os
from application_default import main
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_main() -> None:
    if False:
        print('Hello World!')
    main(PROJECT)