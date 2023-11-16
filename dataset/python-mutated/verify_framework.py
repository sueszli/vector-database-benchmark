""" SeleniumBase Verification """
if __name__ == '__main__':
    from pytest import main
    main([__file__, '-v', '-s'])

def test_simple_cases(pytester):
    if False:
        i = 10
        return i + 15
    'Verify a simple passing test and a simple failing test.\n    The failing test is marked as xfail to have it skipped.'
    pytester.makepyfile("\n        import pytest\n        from seleniumbase import BaseCase\n        class MyTestCase(BaseCase):\n            def test_passing(self):\n                self.assert_equal('yes', 'yes')\n            @pytest.mark.xfail\n            def test_failing(self):\n                self.assert_equal('yes', 'no')\n        ")
    result = pytester.inline_run('--headless', '--rs', '-v')
    assert result.matchreport('test_passing').passed
    assert result.matchreport('test_failing').skipped

def test_basecase(pytester):
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        from seleniumbase import BaseCase\n        class MyTest(BaseCase):\n            def test_basecase(self):\n                self.open("data:text/html,<p>Hello<br><input></p>")\n                self.assert_element("html > body")  # selector\n                self.assert_text("Hello", "body p")  # text, selector\n                self.type("input", "Goodbye")  # selector, text\n                self.click("body p")  # selector\n        ')
    result = pytester.inline_run('--headless', '-v')
    assert result.matchreport('test_basecase').passed

def test_run_with_dashboard(pytester):
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        from seleniumbase import BaseCase\n        class MyTestCase(BaseCase):\n            def test_1_passing(self):\n                self.assert_equal(\'yes\', \'yes\')\n            def test_2_failing(self):\n                self.assert_equal(\'yes\', \'no\')\n            def test_3_skipped(self):\n                self.skip("Skip!")\n        ')
    result = pytester.inline_run('--headless', '--rs', '--dashboard', '-v')
    assert result.matchreport('test_1_passing').passed
    assert result.matchreport('test_2_failing').failed
    assert result.matchreport('test_3_skipped').skipped

def test_sb_fixture(pytester):
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        def test_sb_fixture(sb):\n            sb.open("data:text/html,<p>Hello<br><input></p>")\n            sb.assert_element("html > body")  # selector\n            sb.assert_text("Hello", "body p")  # text, selector\n            sb.type("input", "Goodbye")  # selector, text\n            sb.click("body p")  # selector\n        ')
    result = pytester.inline_run('--headless', '-v')
    assert result.matchreport('test_sb_fixture').passed

def test_request_sb_fixture(pytester):
    if False:
        i = 10
        return i + 15
    pytester.makepyfile('\n        def test_request_sb_fixture(request):\n            sb = request.getfixturevalue(\'sb\')\n            sb.open("data:text/html,<p>Hello<br><input></p>")\n            sb.assert_element("html > body")  # selector\n            sb.assert_text("Hello", "body p")  # text, selector\n            sb.type("input", "Goodbye")  # selector, text\n            sb.click("body p")  # selector\n            sb.tearDown()\n        ')
    result = pytester.inline_run('--headless', '-v')
    assert result.matchreport('test_request_sb_fixture').passed

def check_outcome_field(outcomes, field_name, expected_value):
    if False:
        print('Hello World!')
    field_value = outcomes.get(field_name, 0)
    assert field_value == expected_value, 'outcomes.%s has an unexpected value! Expected "%s" but got "%s"!' % (field_name, expected_value, field_value)

def assert_outcomes(result, passed=1, skipped=0, failed=0, xfailed=0, xpassed=0, rerun=0):
    if False:
        print('Hello World!')
    outcomes = result.parseoutcomes()
    check_outcome_field(outcomes, 'passed', passed)
    check_outcome_field(outcomes, 'skipped', skipped)
    check_outcome_field(outcomes, 'failed', failed)
    check_outcome_field(outcomes, 'xfailed', xfailed)
    check_outcome_field(outcomes, 'xpassed', xpassed)
    check_outcome_field(outcomes, 'rerun', rerun)

def test_rerun_failures(pytester):
    if False:
        for i in range(10):
            print('nop')
    pytester.makepyfile("\n        from seleniumbase import BaseCase\n        class MyTestCase(BaseCase):\n            def test_passing(self):\n                self.assert_equal('yes', 'yes')\n            def test_failing(self):\n                self.assert_equal('yes', 'no')\n        ")
    result = pytester.runpytest('--headless', '--reruns=1', '--rs', '-v')
    assert_outcomes(result, passed=1, failed=1, rerun=1)

def test_browser_launcher(pytester):
    if False:
        print('Hello World!')
    pytester.makepyfile('\n        from seleniumbase import get_driver\n        def test_browser_launcher():\n            success = False\n            try:\n                driver = get_driver("chrome", headless=True)\n                driver.get("data:text/html,<p>Data URL</p>")\n                source = driver.page_source\n                assert "Data URL" in source\n                success = True  # No errors\n            finally:\n                driver.quit()\n            assert success\n        ')
    result = pytester.inline_run('--headless', '-v')
    assert result.matchreport('test_browser_launcher').passed

def test_framework_components(pytester):
    if False:
        return 10
    pytester.makepyfile('\n        from seleniumbase import get_driver\n        from seleniumbase import js_utils\n        from seleniumbase import page_actions\n        def test_framework_components():\n            success = False\n            try:\n                driver = get_driver("chrome", headless=True)\n                driver.get(\'data:text/html,<h1 class="top">Data URL</h2>\')\n                source = driver.page_source\n                assert "Data URL" in source\n                assert page_actions.is_element_visible(driver, "h1.top")\n                js_utils.highlight_with_js(driver, "h1.top", 2, "")\n                success = True  # No errors\n            finally:\n                driver.quit()\n            assert success\n        ')
    result = pytester.inline_run('--headless', '-v', '-s')
    assert result.matchreport('test_framework_components').passed