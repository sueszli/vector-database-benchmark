from __future__ import annotations
import sys
import openai
from .. import OpenAI, _load_client
from .._compat import model_json
from .._models import BaseModel

class Colors:
    HEADER = '\x1b[95m'
    OKBLUE = '\x1b[94m'
    OKGREEN = '\x1b[92m'
    WARNING = '\x1b[93m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'
    BOLD = '\x1b[1m'
    UNDERLINE = '\x1b[4m'

def get_client() -> OpenAI:
    if False:
        print('Hello World!')
    return _load_client()

def organization_info() -> str:
    if False:
        return 10
    organization = openai.organization
    if organization is not None:
        return '[organization={}] '.format(organization)
    return ''

def print_model(model: BaseModel) -> None:
    if False:
        i = 10
        return i + 15
    sys.stdout.write(model_json(model, indent=2) + '\n')

def can_use_http2() -> bool:
    if False:
        print('Hello World!')
    try:
        import h2
    except ImportError:
        return False
    return True