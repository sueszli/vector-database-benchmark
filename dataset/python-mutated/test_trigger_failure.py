def failure_test_trigger():
    if False:
        while True:
            i = 10
    raise RuntimeError('This test is intended to always fail to allow for verification of CI scripting.')