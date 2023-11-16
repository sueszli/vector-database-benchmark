"""Manually verify pages quickly while assisted by automation."""
import os
import shutil
import sys
import time
from selenium.common.exceptions import NoAlertPresentException
from selenium.common.exceptions import WebDriverException
from seleniumbase import BaseCase
from seleniumbase.core.style_sheet import get_report_style
from seleniumbase.config import settings
from seleniumbase.fixtures import js_utils

class MasterQA(BaseCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.check_count = 0
        self.auto_close_results_page = False
        super().setUp(masterqa_mode=True)
        self.LATEST_REPORT_DIR = settings.LATEST_REPORT_DIR
        self.ARCHIVE_DIR = settings.REPORT_ARCHIVE_DIR
        self.RESULTS_PAGE = settings.HTML_REPORT
        self.BAD_PAGE_LOG = settings.RESULTS_TABLE
        self.DEFAULT_VALIDATION_TITLE = 'Manual Check'
        self.DEFAULT_VALIDATION_MESSAGE = settings.MASTERQA_DEFAULT_VALIDATION_MESSAGE
        self.WAIT_TIME_BEFORE_VERIFY = settings.MASTERQA_WAIT_TIME_BEFORE_VERIFY
        self.START_IN_FULL_SCREEN_MODE = settings.MASTERQA_START_IN_FULL_SCREEN_MODE
        self.MAX_IDLE_TIME_BEFORE_QUIT = settings.MASTERQA_MAX_IDLE_TIME_BEFORE_QUIT
        self.__manual_check_setup()
        if self.headless:
            self.auto_close_results_page = True
        if self.START_IN_FULL_SCREEN_MODE:
            self.maximize_window()

    def verify(self, *args):
        if False:
            for i in range(10):
                print('nop')
        warn_msg = '\nWARNING: MasterQA skips manual checks in headless mode!'
        self.check_count += 1
        if self.headless:
            if self.check_count == 1:
                print(warn_msg)
            return
        self.__manual_page_check(*args)

    def auto_close_results(self):
        if False:
            return 10
        'If this method is called, the results page will automatically close\n        at the end of the test run, rather than waiting on the user to close\n        the results page manually.\n        '
        self.auto_close_results_page = True

    def tearDown(self):
        if False:
            return 10
        if self.headless and self.check_count > 0:
            print('WARNING: %s manual checks were skipped! (MasterQA)' % self.check_count)
        if self.__has_exception():
            self.__add_failure(sys.exc_info()[1])
        self.__process_manual_check_results(self.auto_close_results_page)
        super().tearDown()

    def __get_timestamp(self):
        if False:
            while True:
                i = 10
        return str(int(time.time() * 1000))

    def __manual_check_setup(self):
        if False:
            print('Hello World!')
        self.manual_check_count = 0
        self.manual_check_successes = 0
        self.incomplete_runs = 0
        self.page_results_list = []
        self.__clear_out_old_logs(archive_past_runs=False)

    def __clear_out_old_logs(self, archive_past_runs=True, get_log_folder=False):
        if False:
            for i in range(10):
                print('nop')
        abs_path = os.path.abspath('.')
        file_path = os.path.join(abs_path, self.LATEST_REPORT_DIR)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if archive_past_runs:
            archive_timestamp = int(time.time())
            archive_dir_root = os.path.join(file_path, '..', self.ARCHIVE_DIR)
            if not os.path.exists(archive_dir_root):
                os.makedirs(archive_dir_root)
            archive_dir = os.path.join(archive_dir_root, 'log_%s' % archive_timestamp)
            shutil.move(file_path, archive_dir)
            os.makedirs(file_path)
            if get_log_folder:
                return archive_dir
        else:
            latest_report_local = os.path.join('.', self.LATEST_REPORT_DIR)
            filelist = [f for f in os.listdir(latest_report_local) if f.startswith('failed_') or f == self.RESULTS_PAGE or f.startswith('automation_failure') or (f == self.BAD_PAGE_LOG)]
            for f in filelist:
                os.remove(os.path.join(file_path, f))

    def __jq_confirm_dialog(self, question):
        if False:
            i = 10
            return i + 15
        count = self.manual_check_count + 1
        title = self.DEFAULT_VALIDATION_TITLE
        title_content = '<center><font color="#7700bb">%s #%s:</font></center><hr><font color="#0066ff">%s</font>' % (title, count, question)
        title_content = js_utils.escape_quotes_if_needed(title_content)
        jqcd = 'jconfirm({\n                    boxWidth: \'32.5%%\',\n                    useBootstrap: false,\n                    containerFluid: false,\n                    animationBounce: 1,\n                    type: \'default\',\n                    theme: \'bootstrap\',\n                    typeAnimated: true,\n                    animation: \'scale\',\n                    draggable: true,\n                    dragWindowGap: 1,\n                    container: \'body\',\n                    title: \'%s\',\n                    content: \'\',\n                    buttons: {\n                        pass_button: {\n                            btnClass: \'btn-green\',\n                            text: \'YES / PASS\',\n                            keys: [\'y\', \'p\', \'1\'],\n                            action: function(){\n                                $jqc_status = "Success!";\n                                jconfirm.lastButtonText = "Success!";\n                            }\n                        },\n                        fail_button: {\n                            btnClass: \'btn-red\',\n                            text: \'NO / FAIL\',\n                            keys: [\'n\', \'f\', \'2\'],\n                            action: function(){\n                                $jqc_status = "Failure!";\n                                jconfirm.lastButtonText = "Failure!";\n                            }\n                        }\n                    }\n                });' % title_content
        self.execute_script(jqcd)

    def __manual_page_check(self, *args):
        if False:
            i = 10
            return i + 15
        if not args:
            instructions = self.DEFAULT_VALIDATION_MESSAGE
        else:
            instructions = str(args[0])
            if len(args) > 1:
                pass
        question = 'Approve?'
        if instructions and '?' not in instructions:
            question = instructions + ' <> Approve?'
        elif instructions and '?' in instructions:
            question = instructions
        wait_time_before_verify = self.WAIT_TIME_BEFORE_VERIFY
        if self.verify_delay:
            wait_time_before_verify = float(self.verify_delay)
        time.sleep(wait_time_before_verify)
        use_jqc = False
        self.wait_for_ready_state_complete()
        if js_utils.is_jquery_confirm_activated(self.driver):
            use_jqc = True
        else:
            js_utils.activate_jquery_confirm(self.driver)
            get_jqc = None
            try:
                get_jqc = self.execute_script('return jconfirm')
                if get_jqc is None:
                    raise Exception('jconfirm did not load')
                use_jqc = True
            except Exception:
                use_jqc = False
        if use_jqc:
            self.__jq_confirm_dialog(question)
            time.sleep(0.02)
            waiting_for_response = True
            while waiting_for_response:
                time.sleep(0.05)
                jqc_open = self.execute_script('return jconfirm.instances.length')
                if str(jqc_open) == '0':
                    break
            time.sleep(0.1)
            status = None
            try:
                status = self.execute_script('return $jqc_status')
            except Exception:
                status = self.execute_script('return jconfirm.lastButtonText')
        else:
            if self.browser == 'ie':
                text = self.execute_script('if(confirm("%s")){return "Success!"}\n                    else{return "Failure!"}' % question)
            elif self.browser == 'chrome':
                self.execute_script('if(confirm("%s"))\n                    {window.master_qa_result="Success!"}\n                    else{window.master_qa_result="Failure!"}' % question)
                time.sleep(0.05)
                self.__wait_for_special_alert_absent()
                text = self.execute_script('return window.master_qa_result')
            else:
                try:
                    self.execute_script('if(confirm("%s"))\n                        {window.master_qa_result="Success!"}\n                        else{window.master_qa_result="Failure!"}' % question)
                except WebDriverException:
                    pass
                time.sleep(0.05)
                self.__wait_for_special_alert_absent()
                text = self.execute_script('return window.master_qa_result')
            status = text
        self.manual_check_count += 1
        try:
            current_url = self.driver.current_url
        except Exception:
            current_url = self.execute_script('return document.URL')
        if 'Success!' in str(status):
            self.manual_check_successes += 1
            self.page_results_list.append('"%s","%s","%s","%s","%s","%s","%s","%s"' % (self.manual_check_count, 'Success', '-', current_url, self.browser, self.__get_timestamp()[:-3], instructions, '*'))
            return 1
        else:
            bad_page_name = 'failed_check_%s.png' % self.manual_check_count
            self.save_screenshot(bad_page_name, folder=self.LATEST_REPORT_DIR)
            self.page_results_list.append('"%s","%s","%s","%s","%s","%s","%s","%s"' % (self.manual_check_count, 'FAILED!', bad_page_name, current_url, self.browser, self.__get_timestamp()[:-3], instructions, '*'))
            return 0

    def __wait_for_special_alert_absent(self):
        if False:
            for i in range(10):
                print('nop')
        timeout = self.MAX_IDLE_TIME_BEFORE_QUIT
        for x in range(int(timeout * 20)):
            try:
                alert = self.driver.switch_to.alert
                dummy_variable = alert.text
                if '?' not in dummy_variable:
                    return
                time.sleep(0.05)
            except NoAlertPresentException:
                return
        self.driver.quit()
        raise Exception('%s seconds passed without human action! Stopping...' % timeout)

    def __has_exception(self):
        if False:
            for i in range(10):
                print('nop')
        has_exception = False
        if hasattr(sys, 'last_traceback') and sys.last_traceback is not None:
            has_exception = True
        elif hasattr(self, '_outcome'):
            if hasattr(self._outcome, 'errors') and self._outcome.errors:
                has_exception = True
        else:
            has_exception = sys.exc_info()[1] is not None
        return has_exception

    def __add_failure(self, exception=None):
        if False:
            for i in range(10):
                print('nop')
        exc_info = None
        if exception:
            if hasattr(exception, 'msg'):
                exc_info = exception.msg
            elif hasattr(exception, 'message'):
                exc_info = exception.message
            else:
                exc_info = '(Unknown Exception)'
        self.incomplete_runs += 1
        error_page = 'automation_failure_%s.png' % self.incomplete_runs
        self.save_screenshot(error_page, folder=self.LATEST_REPORT_DIR)
        self.page_results_list.append('"%s","%s","%s","%s","%s","%s","%s","%s"' % ('ERR', 'ERROR!', error_page, self.driver.current_url, self.browser, self.__get_timestamp()[:-3], '-', exc_info))
        try:
            self.driver.switch_to_window(self.driver.window_handles[1])
            self.driver.close()
            self.driver.switch_to_window(self.driver.window_handles[0])
        except Exception:
            pass

    def __add_bad_page_log_file(self):
        if False:
            i = 10
            return i + 15
        abs_path = os.path.abspath('.')
        file_path = os.path.join(abs_path, self.LATEST_REPORT_DIR)
        log_file = os.path.join(file_path, self.BAD_PAGE_LOG)
        f = open(log_file, 'w')
        h_p1 = '"Num","Result","Screenshot","URL","Browser","Epoch Time",'
        h_p2 = '"Verification Instructions","Additional Info"\n'
        page_header = h_p1 + h_p2
        f.write(page_header)
        for line in self.page_results_list:
            f.write('%s\n' % line)
        f.close()

    def __add_results_page(self, html):
        if False:
            for i in range(10):
                print('nop')
        abs_path = os.path.abspath('.')
        file_path = os.path.join(abs_path, self.LATEST_REPORT_DIR)
        results_file_name = self.RESULTS_PAGE
        results_file = os.path.join(file_path, results_file_name)
        f = open(results_file, 'w')
        f.write(html)
        f.close()
        return results_file

    def __process_manual_check_results(self, auto_close_results_page=False):
        if False:
            i = 10
            return i + 15
        perfection = True
        failures_count = self.manual_check_count - self.manual_check_successes
        if not self.headless:
            print('')
        print('\n*** MasterQA Manual Test Results: ***')
        if self.manual_check_successes == self.manual_check_count:
            pass
        else:
            print('WARNING: Not all tests passed manual inspection!')
            perfection = False
        if self.incomplete_runs > 0:
            print('WARNING: Not all tests finished running!')
            perfection = False
        if perfection:
            if self.manual_check_count > 0:
                print('SUCCESS: Everything checks out OKAY!')
            else:
                print('WARNING: No manual checks were performed!')
        else:
            pass
        self.__add_bad_page_log_file()
        log_string = self.__clear_out_old_logs(get_log_folder=True)
        log_folder = log_string.split(os.sep)[-1]
        abs_path = os.path.abspath('.')
        file_path = os.path.join(abs_path, self.ARCHIVE_DIR)
        log_path = os.path.join(file_path, log_folder)
        web_log_path = 'file://%s' % log_path
        tf_color = '#11BB11'
        if failures_count > 0:
            tf_color = '#EE3A3A'
        ir_color = '#11BB11'
        if self.incomplete_runs > 0:
            ir_color = '#EE3A3A'
        summary_table = '<div><table><thead><tr>\n              <th>TESTING SUMMARY</th>\n              <th>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>\n              </tr></thead><tbody>\n              <tr style="color:#00BB00"><td>CHECKS PASSED: <td>%s</tr>\n              <tr style="color:%s"     ><td>CHECKS FAILED: <td>%s</tr>\n              <tr style="color:#4D4DDD"><td>TOTAL VERIFICATIONS: <td>%s</tr>\n              <tr style="color:%s"     ><td>INCOMPLETE TEST RUNS: <td>%s</tr>\n              </tbody></table>' % (self.manual_check_successes, tf_color, failures_count, self.manual_check_count, ir_color, self.incomplete_runs)
        summary_table = '<h1 id="ContextHeader" class="sectionHeader" title="">\n                     %s</h1>' % summary_table
        log_link_shown = os.path.join('..', '%s%s' % (self.ARCHIVE_DIR, web_log_path.split(self.ARCHIVE_DIR)[1]))
        csv_link = os.path.join(web_log_path, self.BAD_PAGE_LOG)
        csv_link_shown = '%s' % self.BAD_PAGE_LOG
        log_table = '<p><p><p><p><h2><table><tbody>\n            <tr><td>LOG FILES LINK:&nbsp;&nbsp;<td><a href="%s">%s</a></tr>\n            <tr><td>RESULTS TABLE:&nbsp;&nbsp;<td><a href="%s">%s</a></tr>\n            </tbody></table></h2><p><p><p><p>' % (web_log_path, log_link_shown, csv_link, csv_link_shown)
        failure_table = '<h2><table><tbody></div>'
        any_screenshots = False
        for line in self.page_results_list:
            line = line.split(',')
            if line[1] == '"FAILED!"' or line[1] == '"ERROR!"':
                if not any_screenshots:
                    any_screenshots = True
                    failure_table += '<thead><tr>\n                        <th>SCREENSHOT FILE&nbsp;&nbsp;&nbsp;&nbsp;</th>\n                        <th>LOCATION OF FAILURE</th>\n                        </tr></thead>'
                display_url = line[3]
                if len(display_url) > 60:
                    display_url = display_url[0:58] + '...'
                line = '<a href="%s">%s</a>' % ('file://' + log_path + '/' + line[2], line[2]) + '\n                    &nbsp;&nbsp;&nbsp;&nbsp;<td>\n                    ' + '<a href="%s">%s</a>' % (line[3], display_url)
                line = line.replace('"', '')
                failure_table += '<tr><td>%s</tr>\n' % line
        failure_table += '</tbody></table>'
        table_view = '%s%s%s' % (summary_table, log_table, failure_table)
        report_html = '<html><head>%s</head><body>%s</body></html>' % (get_report_style(), table_view)
        results_file = self.__add_results_page(report_html)
        archived_results_file = os.path.join(log_path, self.RESULTS_PAGE)
        shutil.copyfile(results_file, os.path.realpath(archived_results_file))
        if self.manual_check_count > 0:
            print('\n*** The manual test report is located at:\n' + results_file)
        self.open('file://%s' % archived_results_file)
        if auto_close_results_page:
            time.sleep(1.0)
        else:
            print('\n*** Close the html report window to continue ***')
            try:
                while len(self.driver.window_handles):
                    time.sleep(0.1)
            except Exception:
                pass