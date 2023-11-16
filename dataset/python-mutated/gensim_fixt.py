def setup_module():
    if False:
        print('Hello World!')
    import pytest
    pytest.importorskip('gensim')