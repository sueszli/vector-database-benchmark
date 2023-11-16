"""
Test the visualiser's javascript using PhantomJS.

"""
import os
import luigi
import subprocess
import sys
import unittest
import time
import threading
from selenium import webdriver
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from server_test import ServerTestBase
TEST_TIMEOUT = 40

@unittest.skipUnless(os.environ.get('TEST_VISUALISER'), 'PhantomJS tests not requested in TEST_VISUALISER')
class TestVisualiser(ServerTestBase):
    """
    Builds a medium-sized task tree of MergeSort results then starts
    phantomjs  as a subprocess to interact with the scheduler.

    """

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestVisualiser, self).setUp()
        x = 'I scream for ice cream'
        task = UberTask(base_task=FailingMergeSort, x=x, copies=4)
        luigi.build([task], workers=1, scheduler_port=self.get_http_port())
        self.done = threading.Event()

        def _do_ioloop():
            if False:
                print('Hello World!')
            print('Entering event loop in separate thread')
            for i in range(TEST_TIMEOUT):
                try:
                    self.wait(timeout=1)
                except AssertionError:
                    pass
                if self.done.is_set():
                    break
            print('Exiting event loop thread')
        self.iothread = threading.Thread(target=_do_ioloop)
        self.iothread.start()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.done.set()
        self.iothread.join()

    def test(self):
        if False:
            print('Hello World!')
        port = self.get_http_port()
        print('Server port is {}'.format(port))
        print('Starting phantomjs')
        p = subprocess.Popen('phantomjs {}/phantomjs_test.js http://localhost:{}'.format(here, port), shell=True, stdin=None)
        status = None
        for x in range(TEST_TIMEOUT):
            status = p.poll()
            if status is not None:
                break
            time.sleep(1)
        if status is None:
            raise AssertionError('PhantomJS failed to complete')
        else:
            print('PhantomJS return status is {}'.format(status))
            assert status == 0

    def test_keeps_entries_after_page_refresh(self):
        if False:
            i = 10
            return i + 15
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        driver.get('http://localhost:{}'.format(port))
        length_select = driver.find_element_by_css_selector('select[name="taskTable_length"]')
        assert length_select.get_attribute('value') == '10'
        assert len(driver.find_elements_by_css_selector('#taskTable tbody tr')) == 10
        clicked = False
        for option in length_select.find_elements_by_css_selector('option'):
            if option.text == '50':
                option.click()
                clicked = True
                break
        assert clicked, 'Could not click option with "50" entries.'
        assert length_select.get_attribute('value') == '50'
        assert len(driver.find_elements_by_css_selector('#taskTable tbody tr')) == 50
        driver.refresh()
        length_select = driver.find_element_by_css_selector('select[name="taskTable_length"]')
        assert length_select.get_attribute('value') == '50'
        assert len(driver.find_elements_by_css_selector('#taskTable tbody tr')) == 50

    def test_keeps_table_filter_after_page_refresh(self):
        if False:
            for i in range(10):
                print('nop')
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        driver.get('http://localhost:{}'.format(port))
        search_input = driver.find_element_by_css_selector('input[type="search"]')
        assert search_input.get_attribute('value') == ''
        assert len(driver.find_elements_by_css_selector('#taskTable tbody tr')) == 10
        search_input.send_keys('ber')
        assert len(driver.find_elements_by_css_selector('#taskTable tbody tr')) == 1
        driver.refresh()
        search_input = driver.find_element_by_css_selector('input[type="search"]')
        assert search_input.get_attribute('value') == 'ber'
        assert len(driver.find_elements_by_css_selector('#taskTable tbody tr')) == 1

    def test_keeps_order_after_page_refresh(self):
        if False:
            print('Hello World!')
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        driver.get('http://localhost:{}'.format(port))
        column = driver.find_elements_by_css_selector('#taskTable thead th')[1]
        column.click()
        table_body = driver.find_element_by_css_selector('#taskTable tbody')
        assert self._get_cell_value(table_body, 0, 1) == 'FailingMergeSort_0'
        column.click()
        assert self._get_cell_value(table_body, 0, 1) == 'UberTask'
        driver.refresh()
        table_body = driver.find_element_by_css_selector('#taskTable tbody')
        assert self._get_cell_value(table_body, 0, 1) == 'UberTask'

    def test_keeps_filter_on_server_after_page_refresh(self):
        if False:
            i = 10
            return i + 15
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        driver.get('http://localhost:{}/static/visualiser/index.html#tab=tasks'.format(port))
        checkbox = driver.find_element_by_css_selector('#serverSideCheckbox')
        assert checkbox.is_selected() is False
        checkbox.click()
        driver.refresh()
        checkbox = driver.find_element_by_css_selector('#serverSideCheckbox')
        assert checkbox.is_selected()

    def test_synchronizes_fields_on_tasks_tab(self):
        if False:
            return 10
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        url = 'http://localhost:{}/static/visualiser/index.html#tab=tasks&length=50&search__search=er&filterOnServer=1&order=1,desc'.format(port)
        driver.get(url)
        length_select = driver.find_element_by_css_selector('select[name="taskTable_length"]')
        assert length_select.get_attribute('value') == '50'
        search_input = driver.find_element_by_css_selector('input[type="search"]')
        assert search_input.get_attribute('value') == 'er'
        assert len(driver.find_elements_by_css_selector('#taskTable tbody tr')) == 50
        table_body = driver.find_element_by_css_selector('#taskTable tbody')
        assert self._get_cell_value(table_body, 0, 1) == 'UberTask'

    def test_keeps_invert_after_page_refresh(self):
        if False:
            print('Hello World!')
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        driver.get('http://localhost:{}/static/visualiser/index.html#tab=graph'.format(port))
        invert_checkbox = driver.find_element_by_css_selector('#invertCheckbox')
        assert invert_checkbox.is_selected() is False
        invert_checkbox.click()
        driver.refresh()
        invert_checkbox = driver.find_element_by_css_selector('#invertCheckbox')
        assert invert_checkbox.is_selected()

    def test_keeps_task_id_after_page_refresh(self):
        if False:
            print('Hello World!')
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        driver.get('http://localhost:{}/static/visualiser/index.html#tab=graph'.format(port))
        task_id_input = driver.find_element_by_css_selector('#js-task-id')
        assert task_id_input.get_attribute('value') == ''
        task_id_input.send_keys('1')
        driver.find_element_by_css_selector('#loadTaskForm button[type=submit]').click()
        driver.refresh()
        task_id_input = driver.find_element_by_css_selector('#js-task-id')
        assert task_id_input.get_attribute('value') == '1'

    def test_keeps_hide_done_after_page_refresh(self):
        if False:
            return 10
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        driver.get('http://localhost:{}/static/visualiser/index.html#tab=graph'.format(port))
        hide_done_checkbox = driver.find_element_by_css_selector('#hideDoneCheckbox')
        assert hide_done_checkbox.is_selected() is False
        hide_done_checkbox.click()
        driver.refresh()
        hide_done_checkbox = driver.find_element_by_css_selector('#hideDoneCheckbox')
        assert hide_done_checkbox.is_selected()

    def test_keeps_visualisation_type_after_page_refresh(self):
        if False:
            for i in range(10):
                print('nop')
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        driver.get('http://localhost:{}/static/visualiser/index.html#tab=graph'.format(port))
        svg_radio = driver.find_element_by_css_selector('input[value=svg]')
        assert svg_radio.is_selected()
        d3_radio = driver.find_element_by_css_selector('input[value=d3]')
        d3_radio.find_element_by_xpath('..').click()
        driver.refresh()
        d3_radio = driver.find_element_by_css_selector('input[value=d3]')
        assert d3_radio.is_selected()

    def test_synchronizes_fields_on_graph_tab(self):
        if False:
            for i in range(10):
                print('nop')
        port = self.get_http_port()
        driver = webdriver.PhantomJS()
        url = 'http://localhost:{}/static/visualiser/index.html#tab=graph&taskId=1&invert=1&hideDone=1&visType=svg'.format(port)
        driver.get(url)
        task_id_input = driver.find_element_by_css_selector('#js-task-id')
        assert task_id_input.get_attribute('value') == '1'
        invert_checkbox = driver.find_element_by_css_selector('#invertCheckbox')
        assert invert_checkbox.is_selected()
        hide_done_checkbox = driver.find_element_by_css_selector('#hideDoneCheckbox')
        assert hide_done_checkbox.is_selected()
        svg_radio = driver.find_element_by_css_selector('input[value=svg]')
        assert svg_radio.get_attribute('checked')

    def _get_cell_value(self, elem, row, column):
        if False:
            i = 10
            return i + 15
        tr = elem.find_elements_by_css_selector('#taskTable tbody tr')[row]
        td = tr.find_elements_by_css_selector('td')[column]
        return td.text

def generate_task_families(task_class, n):
    if False:
        print('Hello World!')
    '\n    Generate n copies of a task with different task_family names.\n\n    :param task_class: a subclass of `luigi.Task`\n    :param n: number of copies of `task_class` to create\n    :return: Dictionary of task_family => task_class\n\n    '
    ret = {}
    for i in range(n):
        class_name = '{}_{}'.format(task_class.task_family, i)
        ret[class_name] = type(class_name, (task_class,), {})
    return ret

class UberTask(luigi.Task):
    """
    A task which depends on n copies of a configurable subclass.

    """
    _done = False
    base_task = luigi.TaskParameter()
    x = luigi.Parameter()
    copies = luigi.IntParameter()

    def requires(self):
        if False:
            return 10
        task_families = generate_task_families(self.base_task, self.copies)
        for class_name in task_families:
            yield task_families[class_name](x=self.x)

    def complete(self):
        if False:
            print('Hello World!')
        return self._done

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self._done = True

def popmin(a, b):
    if False:
        while True:
            i = 10
    "\n    popmin(a, b) -> (i, a', b')\n\n    where i is min(a[0], b[0]) and a'/b' are the results of removing i from the\n    relevant sequence.\n    "
    if len(a) == 0:
        return (b[0], a, b[1:])
    elif len(b) == 0:
        return (a[0], a[1:], b)
    elif a[0] > b[0]:
        return (b[0], a, b[1:])
    else:
        return (a[0], a[1:], b)

class MemoryTarget(luigi.Target):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.box = None

    def exists(self):
        if False:
            for i in range(10):
                print('nop')
        return self.box is not None

class MergeSort(luigi.Task):
    x = luigi.Parameter(description='A string to be sorted')

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(MergeSort, self).__init__(*args, **kwargs)
        self.result = MemoryTarget()

    def requires(self):
        if False:
            while True:
                i = 10
        cls = self.__class__
        if len(self.x) > 1:
            p = len(self.x) // 2
            return [cls(self.x[:p]), cls(self.x[p:])]

    def output(self):
        if False:
            print('Hello World!')
        return self.result

    def run(self):
        if False:
            i = 10
            return i + 15
        if len(self.x) > 1:
            (list_1, list_2) = (x.box for x in self.input())
            s = []
            while list_1 or list_2:
                (item, list_1, list_2) = popmin(list_1, list_2)
                s.append(item)
        else:
            s = self.x
        self.result.box = ''.join(s)

class FailingMergeSort(MergeSort):
    """
    Simply fail if the string to sort starts with ' '.

    """
    fail_probability = luigi.FloatParameter(default=0.0)

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.x[0] == ' ':
            raise Exception('I failed')
        else:
            return super(FailingMergeSort, self).run()
if __name__ == '__main__':
    x = 'I scream for ice cream'
    task = UberTask(base_task=FailingMergeSort, x=x, copies=4)
    luigi.build([task], workers=1, scheduler_port=8082)