def test_unique_clears(testdir):
    if False:
        i = 10
        return i + 15
    'Successive uses of the `faker` pytest fixture have the\n    generated unique values cleared between functions.'
    testdir.makepyfile('\n        import pytest\n        from faker.exceptions import UniquenessException\n\n        NUM_SAMPLES = 100\n\n        def test_fully_exhaust_unique_booleans(faker):\n            _dummy = [faker.boolean() for _ in range(NUM_SAMPLES)]\n\n            faker.unique.boolean()\n            faker.unique.boolean()\n            with pytest.raises(UniquenessException):\n                faker.unique.boolean()\n            _dummy = [faker.boolean() for _ in range(NUM_SAMPLES)]\n\n        def test_do_not_exhaust_booleans(faker):\n            faker.unique.boolean()\n\n        def test_fully_exhaust_unique_booleans_again(faker):\n            _dummy = [faker.boolean() for _ in range(NUM_SAMPLES)]\n\n            faker.unique.boolean()\n            faker.unique.boolean()\n            with pytest.raises(UniquenessException):\n                faker.unique.boolean()\n            _dummy = [faker.boolean() for _ in range(NUM_SAMPLES)]\n        ')
    result = testdir.runpytest()
    result.assert_outcomes(passed=3)