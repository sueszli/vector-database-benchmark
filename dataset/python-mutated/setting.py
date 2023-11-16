import os
from enum import Enum
from typing import Dict, List, Set
HOME_DIR = os.environ['HOME']
TOOLS_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir)
PROFILE_DIR = os.path.join(TOOLS_FOLDER, 'profile')
JSON_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, 'json')
MERGED_FOLDER_BASE_DIR = os.path.join(PROFILE_DIR, 'merged')
SUMMARY_FOLDER_DIR = os.path.join(PROFILE_DIR, 'summary')
LOG_DIR = os.path.join(PROFILE_DIR, 'log')

class TestType(Enum):
    CPP: str = 'cxx_test'
    PY: str = 'python_test'

class Test:
    name: str
    target_pattern: str
    test_set: str
    test_type: TestType

    def __init__(self, name: str, target_pattern: str, test_set: str, test_type: TestType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.target_pattern = target_pattern
        self.test_set = test_set
        self.test_type = test_type
TestList = List[Test]
TestStatusType = Dict[str, Set[str]]

class Option:
    need_build: bool = False
    need_run: bool = False
    need_merge: bool = False
    need_export: bool = False
    need_summary: bool = False
    need_pytest: bool = False

class TestPlatform(Enum):
    FBCODE: str = 'fbcode'
    OSS: str = 'oss'

class CompilerType(Enum):
    CLANG: str = 'clang'
    GCC: str = 'gcc'