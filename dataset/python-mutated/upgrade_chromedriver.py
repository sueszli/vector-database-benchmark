"""This script installs the chromedriver version that matches your Chrome.
On newer versions of Python, you may replace "testdir" with "pytester".
(Run with "pytest")"""
import subprocess

class TestUpgradeChromedriver:

    def basic_run(self, testdir):
        if False:
            return 10
        testdir.makepyfile('\n            from seleniumbase import BaseCase\n            class MyTestCase(BaseCase):\n                def test_passing(self):\n                    pass\n            ')
        return testdir

    def upgrade_chromedriver(self, testdir):
        if False:
            for i in range(10):
                print('nop')
        testdir.makepyfile('\n            import subprocess\n            from seleniumbase import BaseCase\n            class MyTestCase(BaseCase):\n                def test_upgrade(self):\n                    chrome_version = self.get_chrome_version()\n                    major_chrome_ver = chrome_version.split(".")[0]\n                    chromedriver_ver = self.get_chromedriver_version()\n                    major_chromedriver_ver = chromedriver_ver.split(".")[0]\n                    if major_chromedriver_ver != major_chrome_ver:\n                        subprocess.check_call(\n                            "sbase get chromedriver %s" % major_chrome_ver,\n                            shell=True\n                        )\n            ')
        return testdir

    def print_versions_of_chromedriver_and_chrome(self, testdir):
        if False:
            print('Hello World!')
        testdir.makepyfile('\n            from seleniumbase import BaseCase\n            class MyTestCase(BaseCase):\n                def test_print_versions(self):\n                    chrome_version = self.get_chrome_version()\n                    major_chrome_ver = chrome_version.split(".")[0]\n                    chromedriver_ver = self.get_chromedriver_version()\n                    major_chromedriver_ver = chromedriver_ver.split(".")[0]\n                    print(\n                        "\\n* Now using chromedriver %s with Chrome %s"\n                        % (chromedriver_ver, chrome_version)\n                    )\n                    if major_chromedriver_ver == major_chrome_ver:\n                        print(\n                            "* SUCCESS: "\n                            "The chromedriver version is compatible "\n                            "with Chrome!"\n                        )\n                    elif major_chromedriver_ver < major_chrome_ver:\n                        print("* !!! Version Mismatch !!!")\n                        print(\n                            "* The version of chromedriver is too low!\\n"\n                            "* Try upgrading to chromedriver %s manually:\\n"\n                            "* >>> sbase get chromedriver %s <<<"\n                            % (major_chrome_ver, major_chrome_ver)\n                        )\n                    else:\n                        print("* !!! Version Mismatch !!!")\n                        print(\n                            "* The version of chromedriver is too high!\\n"\n                            "* Try downgrading to chromedriver %s manually:\\n"\n                            "* >>> sbase get chromedriver %s <<<"\n                            % (major_chrome_ver, major_chrome_ver)\n                        )\n            ')
        return testdir

    def test_upgrade_chromedriver(self, testdir):
        if False:
            for i in range(10):
                print('nop')
        subprocess.check_call('seleniumbase get chromedriver', shell=True)
        testdir = self.basic_run(testdir)
        result = testdir.inline_run('--headless', '-s')
        try:
            assert result.matchreport('test_passing').passed
        except Exception:
            install_command = 'seleniumbase get chromedriver 72.0.3626.69'
            subprocess.check_call(install_command, shell=True)
            testdir = self.upgrade_chromedriver(testdir)
            testdir.inline_run('--headless', '-s')
        testdir = self.print_versions_of_chromedriver_and_chrome(testdir)
        testdir.inline_run('--headless', '-s')
if __name__ == '__main__':
    from pytest import main
    main([__file__])