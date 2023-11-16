"""Unit tests for jobs.job_options."""
from __future__ import annotations
from core.jobs import job_options
from core.tests import test_utils

class JobOptionsTests(test_utils.TestBase):

    def test_default_values(self) -> None:
        if False:
            return 10
        options = job_options.JobOptions()
        self.assertIsNone(options.namespace)

    def test_overwritten_values(self) -> None:
        if False:
            print('Hello World!')
        options = job_options.JobOptions(namespace='abc')
        self.assertEqual(options.namespace, 'abc')

    def test_unsupported_values(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'Unsupported option\\(s\\)'):
            job_options.JobOptions(a='a', b='b')