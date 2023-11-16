"""PageSource Plugin for SeleniumBase tests that run with pynose / nosetests"""
import os
import codecs
from nose.plugins import Plugin
from seleniumbase.config import settings
from seleniumbase.core import log_helper

class PageSource(Plugin):
    """Capture the page source after a test fails."""
    name = 'page_source'
    logfile_name = settings.PAGE_SOURCE_NAME

    def options(self, parser, env):
        if False:
            i = 10
            return i + 15
        super().options(parser, env=env)

    def configure(self, options, conf):
        if False:
            return 10
        super().configure(options, conf)
        if not self.enabled:
            return
        self.options = options

    def addError(self, test, err, capt=None):
        if False:
            for i in range(10):
                print('nop')
        try:
            page_source = test.driver.page_source
        except Exception:
            return
        test_logpath = self.options.log_path + '/' + test.id()
        if not os.path.exists(test_logpath):
            os.makedirs(test_logpath)
        html_file_name = os.path.join(test_logpath, self.logfile_name)
        html_file = codecs.open(html_file_name, 'w+', 'utf-8')
        rendered_source = log_helper.get_html_source_with_base_href(test.driver, page_source)
        html_file.write(rendered_source)
        html_file.close()

    def addFailure(self, test, err, capt=None, tbinfo=None):
        if False:
            i = 10
            return i + 15
        try:
            page_source = test.driver.page_source
        except Exception:
            return
        test_logpath = self.options.log_path + '/' + test.id()
        if not os.path.exists(test_logpath):
            os.makedirs(test_logpath)
        html_file_name = os.path.join(test_logpath, self.logfile_name)
        html_file = codecs.open(html_file_name, 'w+', 'utf-8')
        rendered_source = log_helper.get_html_source_with_base_href(test.driver, page_source)
        html_file.write(rendered_source)
        html_file.close()