from pprint import pformat
import yaml.parser
import yaml.scanner
import yaml
from awxkit.utils import args_string_to_list, seconds_since_date_string
from awxkit.api.resources import resources
from awxkit.api.mixins import HasStatus
import awxkit.exceptions as exc
from . import base
from . import page

class UnifiedJob(HasStatus, base.Base):
    """Base class for unified job pages (e.g. project_updates, inventory_updates
    and jobs).
    """

    def __str__(self):
        if False:
            print('Hello World!')
        items = ['id', 'name', 'status', 'failed', 'result_stdout', 'result_traceback', 'job_explanation', 'job_args']
        info = []
        for item in [x for x in items if hasattr(self, x)]:
            info.append('{0}:{1}'.format(item, getattr(self, item)))
        output = '<{0.__class__.__name__} {1}>'.format(self, ', '.join(info))
        return output.replace('%', '%%')

    @property
    def result_stdout(self):
        if False:
            i = 10
            return i + 15
        if 'result_stdout' not in self.json and 'stdout' in self.related:
            return self.connection.get(self.related.stdout, query_parameters=dict(format='txt_download')).content.decode()
        return self.json.result_stdout.decode()

    def assert_text_in_stdout(self, expected_text, replace_spaces=None, replace_newlines=' '):
        if False:
            while True:
                i = 10
        "Assert text is found in stdout, and if not raise exception with entire stdout.\n\n        Default behavior is to replace newline characters with a space, but this can be modified, including replacement\n        with ''. Pass replace_newlines=None to disable.\n\n        Additionally, you may replace any  with another character (including ''). This is applied after the newline\n        replacement. Default behavior is to not replace spaces.\n        "
        self.wait_until_completed()
        stdout = self.result_stdout
        if replace_newlines is not None:
            stdout = replace_newlines.join([line.strip() for line in stdout.split('\n')])
        if replace_spaces is not None:
            stdout = stdout.replace(' ', replace_spaces)
        if expected_text not in stdout:
            pretty_stdout = pformat(stdout)
            raise AssertionError('Expected "{}", but it was not found in stdout. Full stdout:\n {}'.format(expected_text, pretty_stdout))

    @property
    def is_successful(self):
        if False:
            for i in range(10):
                print('nop')
        "Return whether the current has completed successfully.\n\n        This means that:\n         * self.status == 'successful'\n         * self.has_traceback == False\n         * self.failed == False\n        "
        return super(UnifiedJob, self).is_successful and (not (self.has_traceback or self.failed))

    def wait_until_status(self, status, interval=1, timeout=60, since_job_created=True, **kwargs):
        if False:
            i = 10
            return i + 15
        if since_job_created:
            timeout = timeout - seconds_since_date_string(self.created)
        return super(UnifiedJob, self).wait_until_status(status, interval, timeout, **kwargs)

    def wait_until_completed(self, interval=5, timeout=60 * 8, since_job_created=True, **kwargs):
        if False:
            while True:
                i = 10
        if since_job_created:
            timeout = timeout - seconds_since_date_string(self.created)
        return super(UnifiedJob, self).wait_until_completed(interval, timeout, **kwargs)

    @property
    def has_traceback(self):
        if False:
            i = 10
            return i + 15
        'Return whether a traceback has been detected in result_traceback'
        try:
            tb = str(self.result_traceback)
        except AttributeError:
            tb = ''
        return 'Traceback' in tb

    def cancel(self):
        if False:
            return 10
        cancel = self.get_related('cancel')
        if not cancel.can_cancel:
            return
        try:
            cancel.post()
        except exc.MethodNotAllowed as e:
            if not any(('not allowed' in field for field in e.msg.values())):
                raise e
        return self.get()

    @property
    def job_args(self):
        if False:
            return 10
        'Helper property to return flattened cmdline arg tokens in a list.\n        Flattens arg strings for rough inclusion checks:\n        ```assert "thing" in unified_job.job_args```\n        ```assert dict(extra_var=extra_var_val) in unified_job.job_args```\n        If you need to ensure the job_args are of awx-provided format use raw unified_job.json.job_args.\n        '

        def attempt_yaml_load(arg):
            if False:
                print('Hello World!')
            try:
                return yaml.safe_load(arg)
            except (yaml.parser.ParserError, yaml.scanner.ScannerError):
                return str(arg)
        args = []
        if not self.json.job_args:
            return ''
        for arg in yaml.safe_load(self.json.job_args):
            try:
                args.append(yaml.safe_load(arg))
            except (yaml.parser.ParserError, yaml.scanner.ScannerError):
                if arg[0] == '@':
                    args.append(attempt_yaml_load(arg))
                elif args[-1] == '-c':
                    args.extend([attempt_yaml_load(item) for item in args_string_to_list(arg)])
                else:
                    raise
        return args

    @property
    def controller_dir(self):
        if False:
            i = 10
            return i + 15
        'Returns the path to the private_data_dir on the controller node for the job\n        This can be used if trying to shell in and inspect the files used by the job\n        Cannot use job_cwd, because that is path inside EE container\n        '
        self.get()
        job_args = self.job_args
        expected_prefix = '/tmp/awx_{}'.format(self.id)
        for (arg1, arg2) in zip(job_args[:-1], job_args[1:]):
            if arg1 == '-v':
                if ':' in arg2:
                    host_loc = arg2.split(':')[0]
                    if host_loc.startswith(expected_prefix):
                        return host_loc
        raise RuntimeError('Could not find a controller private_data_dir for this job. Searched for volume mount to {} inside of args {}'.format(expected_prefix, job_args))

class UnifiedJobs(page.PageList, UnifiedJob):
    pass
page.register_page([resources.unified_jobs, resources.instance_related_jobs, resources.instance_group_related_jobs, resources.schedules_jobs], UnifiedJobs)