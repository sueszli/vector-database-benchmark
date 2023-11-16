import argparse
import platform
import sys

class ScaleneArguments(argparse.Namespace):
    """Encapsulates all arguments and default values for Scalene."""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.cpu = True
        self.gpu = platform.system() != 'Darwin'
        self.memory = sys.platform != 'win32'
        self.stacks = False
        self.cpu_percent_threshold = 1
        self.cpu_sampling_rate = 0.01
        self.allocation_sampling_window = 10485767
        self.html = False
        self.json = False
        self.column_width = 132
        self.malloc_threshold = 100
        self.outfile = None
        self.pid = 0
        self.profile_all = False
        self.profile_interval = float('inf')
        self.profile_only = ''
        self.profile_exclude = ''
        self.program_path = ''
        self.reduced_profile = False
        self.use_virtual_time = False
        self.memory_leak_detector = True
        self.web = True
        self.no_browser = False
        self.port = 8088
        self.cli = False