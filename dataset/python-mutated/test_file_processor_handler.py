from __future__ import annotations
import os
import shutil
from datetime import timedelta
import time_machine
from airflow.utils import timezone
from airflow.utils.log.file_processor_handler import FileProcessorHandler

class TestFileProcessorHandler:

    def setup_method(self):
        if False:
            return 10
        self.base_log_folder = '/tmp/log_test'
        self.filename = '{filename}'
        self.filename_template = '{{ filename }}.log'
        self.dag_dir = '/dags'

    def test_non_template(self):
        if False:
            while True:
                i = 10
        date = timezone.utcnow().strftime('%Y-%m-%d')
        handler = FileProcessorHandler(base_log_folder=self.base_log_folder, filename_template=self.filename)
        handler.dag_dir = self.dag_dir
        path = os.path.join(self.base_log_folder, 'latest')
        assert os.path.islink(path)
        assert os.path.basename(os.readlink(path)) == date
        handler.set_context(filename=os.path.join(self.dag_dir, 'logfile'))
        assert os.path.exists(os.path.join(path, 'logfile'))

    def test_template(self):
        if False:
            for i in range(10):
                print('nop')
        date = timezone.utcnow().strftime('%Y-%m-%d')
        handler = FileProcessorHandler(base_log_folder=self.base_log_folder, filename_template=self.filename_template)
        handler.dag_dir = self.dag_dir
        path = os.path.join(self.base_log_folder, 'latest')
        assert os.path.islink(path)
        assert os.path.basename(os.readlink(path)) == date
        handler.set_context(filename=os.path.join(self.dag_dir, 'logfile'))
        assert os.path.exists(os.path.join(path, 'logfile.log'))

    def test_symlink_latest_log_directory(self):
        if False:
            i = 10
            return i + 15
        handler = FileProcessorHandler(base_log_folder=self.base_log_folder, filename_template=self.filename)
        handler.dag_dir = self.dag_dir
        date1 = (timezone.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')
        date2 = (timezone.utcnow() + timedelta(days=2)).strftime('%Y-%m-%d')
        path1 = os.path.join(self.base_log_folder, date1, 'log1')
        path2 = os.path.join(self.base_log_folder, date1, 'log2')
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)
        link = os.path.join(self.base_log_folder, 'latest')
        with time_machine.travel(date1, tick=False):
            handler.set_context(filename=os.path.join(self.dag_dir, 'log1'))
            assert os.path.islink(link)
            assert os.path.basename(os.readlink(link)) == date1
            assert os.path.exists(os.path.join(link, 'log1'))
        with time_machine.travel(date2, tick=False):
            handler.set_context(filename=os.path.join(self.dag_dir, 'log2'))
            assert os.path.islink(link)
            assert os.path.basename(os.readlink(link)) == date2
            assert os.path.exists(os.path.join(link, 'log2'))

    def test_symlink_latest_log_directory_exists(self):
        if False:
            while True:
                i = 10
        handler = FileProcessorHandler(base_log_folder=self.base_log_folder, filename_template=self.filename)
        handler.dag_dir = self.dag_dir
        date1 = (timezone.utcnow() + timedelta(days=1)).strftime('%Y-%m-%d')
        path1 = os.path.join(self.base_log_folder, date1, 'log1')
        if os.path.exists(path1):
            os.remove(path1)
        link = os.path.join(self.base_log_folder, 'latest')
        if os.path.exists(link):
            os.remove(link)
        os.makedirs(link)
        with time_machine.travel(date1, tick=False):
            handler.set_context(filename=os.path.join(self.dag_dir, 'log1'))

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.base_log_folder, ignore_errors=True)