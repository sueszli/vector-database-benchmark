def setup_module():
    if False:
        for i in range(10):
            print('nop')
    import pytest
    from nltk.parse.malt import MaltParser
    try:
        depparser = MaltParser()
    except (AssertionError, LookupError) as e:
        pytest.skip('MaltParser is not available')