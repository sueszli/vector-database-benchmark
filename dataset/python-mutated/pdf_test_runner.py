"""
A test runner is a component which orchestrates the execution of tests and provides the outcome to the user.
The runner may use a graphical interface, a textual interface,
or return a special value to indicate the results of executing the tests.

This class provides a convenience method to add a hook to the unittest framework.
This hook will ensure the output of the tests is exported to PDF.
"""
import datetime
import logging
import subprocess
import time
import typing
import unittest
from pathlib import Path
from borb.pdf.document.document import Document
from borb.pdf.pdf import PDF
from borb.toolkit.test_util.default_test_renderer import DefaultTestRenderer
from borb.toolkit.test_util.test_info import TestResult
from borb.toolkit.test_util.test_renderer import TestRenderer
from borb.toolkit.test_util.test_status import TestStatus
logger = logging.getLogger(__name__)

class PDFTestRunner:
    """
    A test runner is a component which orchestrates the execution of tests and provides the outcome to the user.
    The runner may use a graphical interface, a textual interface,
    or return a special value to indicate the results of executing the tests.

    This class provides a convenience method to add a hook to the unittest framework.
    This hook will ensure the output of the tests is exported to PDF.
    """
    _is_initialized: bool = False
    _test_id_to_start_time: typing.Dict[int, float] = {}
    _test_statuses: typing.List['TestResult'] = []

    @staticmethod
    def _add_test_info(test_case: unittest.TestCase, status: TestStatus) -> None:
        if False:
            while True:
                i = 10
        test_case_started_at: float = min(PDFTestRunner._test_id_to_start_time.values())
        if id(test_case) in PDFTestRunner._test_id_to_start_time:
            test_case_started_at = PDFTestRunner._test_id_to_start_time[id(test_case)]
        PDFTestRunner._test_statuses.append(TestResult(file=test_case.__module__, method=test_case._testMethodName, class_name=test_case.__class__.__name__, started_at=test_case_started_at, status=status, stopped_at=time.time()))

    @staticmethod
    def _build_pdf(renderer: TestRenderer, report_name: Path, open_when_finished: bool) -> None:
        if False:
            while True:
                i = 10
        PDFTestRunner._test_statuses.sort(key=lambda x: x.get_file() + '/' + x.get_class_name() + '/' + x.get_method())
        logger.debug('creating empty Document')
        doc: Document = Document()
        logger.debug('adding (front) cover page(s) to Document')
        renderer.build_pdf_front_cover_page(doc)
        logger.debug('adding summary Page(s) to Document')
        renderer.build_pdf_summary_page(doc, PDFTestRunner._test_statuses)
        file_name_sorted: typing.List[str] = sorted([x for x in set([x.get_file() for x in PDFTestRunner._test_statuses])])
        for (i, class_name) in enumerate(file_name_sorted):
            logger.debug('building class level results %d/%d' % (i + 1, len(file_name_sorted)))
            renderer.build_pdf_module_page(doc, [x for x in PDFTestRunner._test_statuses if x.get_file() == class_name])
        logger.debug('adding (back) cover page(s) to Document')
        renderer.build_pdf_back_cover_page(doc)
        logger.debug('writing PDF to file')
        with open(report_name, 'wb') as fh:
            PDF.dumps(fh, doc)
        if open_when_finished:
            logger.debug('opening PDF')
            subprocess.call(('xdg-open', report_name))

    @staticmethod
    def set_up(test_case: unittest.TestCase, report_name: Path=Path('Test Report %s.pdf' % datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'))):
        if False:
            while True:
                i = 10
        '\n        This function sinks its hooks into the unittest framework and ensures\n        every test (result) is captured and its results can be output to PDF.\n        :param test_case:       the unittest.TestCase that provides the entrypoint into the unittest code\n        :param report_name:     the Path determining where to write the output PDF\n        :return:            None\n        '
        test_result: typing.Optional[unittest.TestResult] = None
        try:
            test_result = test_case._outcome.result
        except:
            pass
        if test_result is None:
            return
        if PDFTestRunner._is_initialized:
            return
        PDFTestRunner._is_initialized = True
        prev_add_error = test_result.addError

        def new_add_error(t: unittest.TestCase, err: typing.Any):
            if False:
                while True:
                    i = 10
            "\n            Called when an error has occurred. 'err' is a tuple of values as\n            returned by sys.exc_info().\n            :param t:\n            :param err:\n            :return:\n            "
            PDFTestRunner._add_test_info(t, TestStatus.ERROR)
            prev_add_error(t, err)
        test_result.addError = new_add_error
        prev_add_expected_failure = test_result.addExpectedFailure

        def new_add_expected_failure(t: unittest.TestCase, err: str):
            if False:
                i = 10
                return i + 15
            '\n            Called when an expected failure/error occurred."\n            :param t:\n            :param err:\n            :return:\n            '
            PDFTestRunner._add_test_info(t, TestStatus.EXPECTED_FAILURE)
            prev_add_expected_failure(t, err)
        test_result.addExpectedFailure = new_add_expected_failure
        prev_add_failure = test_result.addFailure

        def new_add_failure(t: unittest.TestCase, err: typing.Any):
            if False:
                while True:
                    i = 10
            "\n            Called when an error has occurred. 'err' is a tuple of values as\n            returned by sys.exc_info().\n            :param t:\n            :param err:\n            :return:\n            "
            PDFTestRunner._add_test_info(t, TestStatus.FAILURE)
            prev_add_failure(t, err)
        test_result.addFailure = new_add_failure
        prev_add_skip = test_result.addSkip

        def new_add_skip(t: unittest.TestCase, r: str):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Called when a test is skipped.\n            :param t:\n            :param r:\n            :return:\n            '
            PDFTestRunner._add_test_info(t, TestStatus.SKIP)
            prev_add_skip(t, r)
        test_result.addSkip = new_add_skip
        prev_add_success = test_result.addSuccess

        def new_add_success(t: unittest.TestCase):
            if False:
                i = 10
                return i + 15
            '\n            Called when a test has completed successfully\n            :param t:\n            :return:\n            '
            PDFTestRunner._add_test_info(t, TestStatus.SUCCESS)
            prev_add_success(t)
        test_result.addSuccess = new_add_success
        prev_add_unexpected_success = test_result.addUnexpectedSuccess

        def new_add_unexpected_success(t: unittest.TestCase):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Called when a test was expected to fail, but succeed.\n            :param t:\n            :return:\n            '
            PDFTestRunner._add_test_info(t, TestStatus.UNEXPECTED_SUCCESS)
            prev_add_unexpected_success(t)
        test_result.addUnexpectedSuccess = new_add_unexpected_success
        prev_start_test = test_result.startTest

        def new_start_test(t: unittest.TestCase):
            if False:
                i = 10
                return i + 15
            '\n            Called when the given test is about to be run\n            :param t:\n            :return:\n            '
            PDFTestRunner._test_id_to_start_time[id(t)] = time.time()
            prev_start_test(t)
        test_result.startTest = new_start_test
        prev_start_test_run = test_result.startTestRun

        def new_start_test_run():
            if False:
                print('Hello World!')
            '\n            Called once before any tests are executed.\n            See startTest for a method called before each test.\n            :return:\n            '
            PDFTestRunner._test_id_to_start_time.clear()
            PDFTestRunner._test_statuses.clear()
            PDFTestRunner._test_id_to_start_time[-1] = time.time()
            prev_start_test_run()
        test_result.startTestRun = new_start_test_run
        prev_stop_test_run = test_result.stopTestRun

        def new_stop_test_run():
            if False:
                print('Hello World!')
            '\n            Called once after all tests are executed.\n            See stopTest for a method called after each test.\n            :return:\n            '
            prev_stop_test_run()
            PDFTestRunner._build_pdf(renderer=DefaultTestRenderer(), report_name=report_name, open_when_finished=True)
        test_result.stopTestRun = new_stop_test_run
        PDFTestRunner._test_id_to_start_time[-1] = time.time()