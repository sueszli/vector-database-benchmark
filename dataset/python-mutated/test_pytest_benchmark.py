def test_pytest_benchmark(selenium):
    if False:
        print('Hello World!')
    selenium.run_js('\n        await pyodide.loadPackage(["pytest-benchmark", "pytest"]);\n        pyodide.FS.mkdir("/tests")\n        pyodide.FS.writeFile("/tests/test_blah.py",\n`\nimport pytest\n\n@pytest.mark.benchmark\ndef test_blah(benchmark):\n    @benchmark\n    def f():\n        for i in range(100_000):\n            pass\n    assert benchmark.stats.stats.min >= 0.000001\n    assert benchmark.stats.stats.max <= 10\n`\n        );\n        pyodide.FS.chdir("/tests");\n        const pytest = pyodide.pyimport("pytest");\n        pytest.main();\n        pytest.destroy();\n        ')
    assert 'benchmark: 1 tests' in selenium.logs
    assert 'Name (time in ms)' in selenium.logs