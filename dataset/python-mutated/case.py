import pytest

class SecurityCase:

    def setup_method(self):
        if False:
            print('Hello World!')
        pytest.importorskip('cryptography')