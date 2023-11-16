import argparse
import contextlib
import os
import sys
from textwrap import dedent
from typing import Any, List, NoReturn, Optional, Tuple
from scalene.find_browser import find_browser
from scalene.scalene_arguments import ScaleneArguments
from scalene.scalene_version import scalene_version, scalene_date
scalene_gui_url = f"file:{os.path.join(os.path.dirname(__file__), 'scalene-gui', 'index.html')}"

class RichArgParser(argparse.ArgumentParser):

    def __init__(self, *args: Any, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        from rich.console import Console
        self.console = Console()
        super().__init__(*args, **kwargs)

    def _print_message(self, message: Optional[str], file: Any=None) -> None:
        if False:
            print('Hello World!')
        if message:
            self.console.print(message)

class StopJupyterExecution(Exception):
    """NOP exception to enable clean exits from within Jupyter notebooks."""

    def _render_traceback_(self) -> None:
        if False:
            while True:
                i = 10
        pass

class ScaleneParseArgs:

    @staticmethod
    def clean_exit(code: object=0) -> NoReturn:
        if False:
            return 10
        'Replacement for sys.exit that exits cleanly from within Jupyter notebooks.'
        raise StopJupyterExecution

    @staticmethod
    def parse_args() -> Tuple[argparse.Namespace, List[str]]:
        if False:
            while True:
                i = 10
        with contextlib.suppress(BaseException):
            from IPython import get_ipython
            if get_ipython():
                sys.exit = ScaleneParseArgs.clean_exit
                sys._exit = ScaleneParseArgs.clean_exit
        defaults = ScaleneArguments()
        usage = dedent(f'[b]Scalene[/b]: a high-precision CPU and memory profiler, version {scalene_version} ({scalene_date})\n[link=https://github.com/plasma-umass/scalene]https://github.com/plasma-umass/scalene[/link]\n\n\ncommand-line:\n  % [b]scalene \\[options] your_program.py \\[--- --your_program_args] [/b]\nor\n  % [b]python3 -m scalene \\[options] your_program.py \\[--- --your_program_args] [/b]\n\nin Jupyter, line mode:\n[b]  %scrun \\[options] statement[/b]\n\nin Jupyter, cell mode:\n[b]  %%scalene \\[options]\n   your code here\n[/b]\n')
        epilog = dedent('When running Scalene in the background, you can suspend/resume profiling\nfor the process ID that Scalene reports. For example:\n\n   % python3 -m scalene [options] yourprogram.py &\n Scalene now profiling process 12345\n   to suspend profiling: python3 -m scalene.profile --off --pid 12345\n   to resume profiling:  python3 -m scalene.profile --on  --pid 12345\n')
        parser = RichArgParser(prog='scalene', description=usage, epilog=epilog if sys.platform != 'win32' else '', formatter_class=argparse.RawTextHelpFormatter, allow_abbrev=False)
        parser.add_argument('--version', dest='version', action='store_const', const=True, help='prints the version number for this release of Scalene and exits')
        parser.add_argument('--column-width', dest='column_width', type=int, default=defaults.column_width, help=f'Column width for profile output (default: [blue]{defaults.column_width}[/blue])')
        parser.add_argument('--outfile', type=str, default=defaults.outfile, help='file to hold profiler output (default: [blue]' + ('stdout' if not defaults.outfile else defaults.outfile) + '[/blue])')
        parser.add_argument('--html', dest='html', action='store_const', const=True, default=defaults.html, help='output as HTML (default: [blue]' + str('html' if defaults.html else 'web') + '[/blue])')
        parser.add_argument('--json', dest='json', action='store_const', const=True, default=defaults.json, help='output as JSON (default: [blue]' + str('json' if defaults.json else 'web') + '[/blue])')
        parser.add_argument('--cli', dest='cli', action='store_const', const=True, default=defaults.cli, help='forces use of the command-line')
        parser.add_argument('--stacks', dest='stacks', action='store_const', const=True, default=defaults.stacks, help='collect stack traces')
        parser.add_argument('--web', dest='web', action='store_const', const=True, default=defaults.web, help="opens a web tab to view the profile (saved as 'profile.html')")
        parser.add_argument('--no-browser', dest='no_browser', action='store_const', const=True, default=defaults.no_browser, help="doesn't open a web tab; just saves the profile ('profile.html')")
        parser.add_argument('--viewer', dest='viewer', action='store_const', const=True, default=False, help=f'opens the Scalene web UI and exits.')
        parser.add_argument('--reduced-profile', dest='reduced_profile', action='store_const', const=True, default=defaults.reduced_profile, help=f'generate a reduced profile, with non-zero lines only (default: [blue]{defaults.reduced_profile}[/blue])')
        parser.add_argument('--profile-interval', type=float, default=defaults.profile_interval, help=f'output profiles every so many seconds (default: [blue]{defaults.profile_interval}[/blue])')
        parser.add_argument('--cpu', dest='cpu', action='store_const', const=True, default=None, help='profile CPU time (default: [blue] True [/blue])')
        parser.add_argument('--cpu-only', dest='cpu', action='store_const', const=True, default=None, help='profile CPU time ([red]deprecated: use --cpu [/red])')
        parser.add_argument('--gpu', dest='gpu', action='store_const', const=True, default=None, help='profile GPU time and memory (default: [blue]' + str(defaults.gpu) + ' [/blue])')
        if sys.platform == 'win32':
            memory_profile_message = 'profile memory (not supported on this platform)'
        else:
            memory_profile_message = 'profile memory (default: [blue]' + str(defaults.memory) + ' [/blue])'
        parser.add_argument('--memory', dest='memory', action='store_const', const=True, default=None, help=memory_profile_message)
        parser.add_argument('--profile-all', dest='profile_all', action='store_const', const=True, default=defaults.profile_all, help='profile all executed code, not just the target program (default: [blue]' + ('all code' if defaults.profile_all else 'only the target program') + '[/blue])')
        parser.add_argument('--profile-only', dest='profile_only', type=str, default=defaults.profile_only, help='profile only code in filenames that contain the given strings, separated by commas (default: [blue]' + ('no restrictions' if not defaults.profile_only else defaults.profile_only) + '[/blue])')
        parser.add_argument('--profile-exclude', dest='profile_exclude', type=str, default=defaults.profile_exclude, help='do not profile code in filenames that contain the given strings, separated by commas (default: [blue]' + ('no restrictions' if not defaults.profile_exclude else defaults.profile_exclude) + '[/blue])')
        parser.add_argument('--use-virtual-time', dest='use_virtual_time', action='store_const', const=True, default=defaults.use_virtual_time, help=f'measure only CPU time, not time spent in I/O or blocking (default: [blue]{defaults.use_virtual_time}[/blue])')
        parser.add_argument('--cpu-percent-threshold', dest='cpu_percent_threshold', type=float, default=defaults.cpu_percent_threshold, help=f'only report profiles with at least this percent of CPU time (default: [blue]{defaults.cpu_percent_threshold}%%[/blue])')
        parser.add_argument('--cpu-sampling-rate', dest='cpu_sampling_rate', type=float, default=defaults.cpu_sampling_rate, help=f'CPU sampling rate (default: every [blue]{defaults.cpu_sampling_rate}s[/blue])')
        parser.add_argument('--allocation-sampling-window', dest='allocation_sampling_window', type=int, default=defaults.allocation_sampling_window, help=f'Allocation sampling window size, in bytes (default: [blue]{defaults.allocation_sampling_window} bytes[/blue])')
        parser.add_argument('--malloc-threshold', dest='malloc_threshold', type=int, default=defaults.malloc_threshold, help=f'only report profiles with at least this many allocations (default: [blue]{defaults.malloc_threshold}[/blue])')
        parser.add_argument('--program-path', dest='program_path', type=str, default='', help='The directory containing the code to profile (default: [blue]the path to the profiled program[/blue])')
        parser.add_argument('--memory-leak-detector', dest='memory_leak_detector', action='store_true', default=defaults.memory_leak_detector, help='EXPERIMENTAL: report likely memory leaks (default: [blue]' + str(defaults.memory_leak_detector) + '[/blue])')
        parser.add_argument('--ipython', dest='ipython', action='store_const', const=True, default=False, help=argparse.SUPPRESS)
        if sys.platform != 'win32':
            group = parser.add_mutually_exclusive_group(required=False)
            group.add_argument('--on', action='store_true', help='start with profiling on (default)')
            group.add_argument('--off', action='store_true', help='start with profiling off')
        parser.add_argument('--pid', type=int, default=0, help=argparse.SUPPRESS)
        parser.add_argument('---', dest='unused_args', default=[], help=argparse.SUPPRESS, nargs=argparse.REMAINDER)
        (args, left) = parser.parse_known_args()
        if sys.platform == 'win32':
            args.on = True
            args.pid = 0
        left += args.unused_args
        import re
        if args.viewer:
            if (browser := find_browser()):
                browser.open(scalene_gui_url)
            else:
                print(f'Scalene: could not open {scalene_gui_url}.')
            sys.exit(0)
        if args.cpu or args.gpu or args.memory:
            if not args.memory:
                args.memory = False
            if not args.gpu:
                args.gpu = False
        else:
            args.cpu = defaults.cpu
            args.gpu = defaults.gpu
            args.memory = defaults.memory
        args.cpu = True
        in_jupyter_notebook = len(sys.argv) >= 1 and re.match('_ipython-input-([0-9]+)-.*', sys.argv[0])
        if not in_jupyter_notebook and len(sys.argv) + len(left) == 1:
            parser.print_help(sys.stderr)
            sys.exit(-1)
        if args.version:
            print(f'Scalene version {scalene_version} ({scalene_date})')
            if not args.ipython:
                sys.exit(-1)
            args = []
        return (args, left)