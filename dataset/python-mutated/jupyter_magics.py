"""A module of magic functions"""
import time
import threading
from IPython import get_ipython
from IPython.display import display
from IPython.core import magic_arguments
from IPython.core.magic import cell_magic, line_magic, Magics, magics_class, register_line_magic
from qiskit.utils import optionals as _optionals
from qiskit.utils.deprecation import deprecate_func
import qiskit
from qiskit.tools.events.progressbar import TextProgressBar
from .progressbar import HTMLProgressBar
from .library import circuit_library_widget

def _html_checker(job_var, interval, status, header, _interval_set=False):
    if False:
        print('Hello World!')
    'Internal function that updates the status\n    of a HTML job monitor.\n\n    Args:\n        job_var (BaseJob): The job to keep track of.\n        interval (int): The status check interval\n        status (widget): HTML ipywidget for output to screen\n        header (str): String representing HTML code for status.\n        _interval_set (bool): Was interval set by user?\n    '
    job_status = job_var.status()
    job_status_name = job_status.name
    job_status_msg = job_status.value
    status.value = header % job_status_msg
    while job_status_name not in ['DONE', 'CANCELLED']:
        time.sleep(interval)
        job_status = job_var.status()
        job_status_name = job_status.name
        job_status_msg = job_status.value
        if job_status_name == 'ERROR':
            break
        if job_status_name == 'QUEUED':
            job_status_msg += ' (%s)' % job_var.queue_position()
            if job_var.queue_position() is None:
                interval = 2
            elif not _interval_set:
                interval = max(job_var.queue_position(), 2)
        elif not _interval_set:
            interval = 2
        status.value = header % job_status_msg
    status.value = header % job_status_msg

@magics_class
class StatusMagic(Magics):
    """A class of status magic functions."""

    @cell_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('-i', '--interval', type=float, default=None, help='Interval for status check.')
    @_optionals.HAS_IPYWIDGETS.require_in_call
    def qiskit_job_status(self, line='', cell=None):
        if False:
            for i in range(10):
                print('nop')
        'A Jupyter magic function to check the status of a Qiskit job instance.'
        import ipywidgets as widgets
        args = magic_arguments.parse_argstring(self.qiskit_job_status, line)
        if args.interval is None:
            args.interval = 2
            _interval_set = False
        else:
            _interval_set = True
        cell_lines = cell.split('\n')
        line_vars = []
        for cline in cell_lines:
            if '=' in cline and '==' not in cline:
                line_vars.append(cline.replace(' ', '').split('=')[0])
            elif '.append(' in cline:
                line_vars.append(cline.replace(' ', '').split('(')[0])
        self.shell.ex(cell)
        jobs = []
        for var in line_vars:
            iter_var = False
            if '#' not in var:
                if '[' in var:
                    var = var.split('[')[0]
                    iter_var = True
                elif '.append' in var:
                    var = var.split('.append')[0]
                    iter_var = True
                if iter_var:
                    for item in self.shell.user_ns[var]:
                        if isinstance(item, qiskit.providers.job.Job):
                            jobs.append(item)
                elif isinstance(self.shell.user_ns[var], qiskit.providers.job.Job):
                    jobs.append(self.shell.user_ns[var])
        if not any(jobs):
            raise Exception('Cell must contain at least one variable of BaseJob type.')
        multi_job = False
        if len(jobs) > 1:
            multi_job = True
        job_checkers = []
        for (idx, job_var) in enumerate(jobs):
            style = 'font-size:16px;'
            if multi_job:
                idx_str = '[%s]' % idx
            else:
                idx_str = ''
            header = f"<p style='{style}'>Job Status {idx_str}: %s </p>"
            status = widgets.HTML(value=header % job_var.status().value)
            thread = threading.Thread(target=_html_checker, args=(job_var, args.interval, status, header, _interval_set))
            thread.start()
            job_checkers.append(status)
        box = widgets.VBox(job_checkers)
        display(box)

@magics_class
class ProgressBarMagic(Magics):
    """A class of progress bar magic functions."""

    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('-t', '--type', type=str, default='html', help="Type of progress bar, 'html' or 'text'.")
    def qiskit_progress_bar(self, line='', cell=None):
        if False:
            i = 10
            return i + 15
        'A Jupyter magic function to generate progressbar.'
        args = magic_arguments.parse_argstring(self.qiskit_progress_bar, line)
        if args.type == 'html':
            pbar = HTMLProgressBar()
        elif args.type == 'text':
            pbar = TextProgressBar()
        else:
            raise qiskit.QiskitError('Invalid progress bar type.')
        return pbar
if _optionals.HAS_MATPLOTLIB and get_ipython():

    @register_line_magic
    @deprecate_func(since='0.25.0', additional_msg='This was originally only for internal documentation and is no longer used.')
    def circuit_library_info(circuit: qiskit.QuantumCircuit) -> None:
        if False:
            print('Hello World!')
        'Displays library information for a quantum circuit.\n\n        Args:\n            circuit: Input quantum circuit.\n        '
        shell = get_ipython()
        circ = shell.ev(circuit)
        circuit_library_widget(circ)