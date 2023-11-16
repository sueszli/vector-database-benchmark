"""
Auto-generate tqdm/completion.sh from docstrings.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import tqdm
import tqdm.cli
RE_OPT = re.compile('(\\w+)  :', flags=re.M)
RE_OPT_INPUT = re.compile('(\\w+)  : (?:str|int|float|chr|dict|tuple)', flags=re.M)

def doc2opt(doc, user_input=True):
    if False:
        while True:
            i = 10
    '\n    doc  : str, document to parse\n    user_input  : bool, optional.\n      [default: True] for only options requiring user input\n    '
    RE = RE_OPT_INPUT if user_input else RE_OPT
    return ('--' + i for i in RE.findall(doc))
options = {'-h', '--help', '-v', '--version'}
options_input = set()
for doc in (tqdm.tqdm.__doc__, tqdm.cli.CLI_EXTRA_DOC):
    options.update(doc2opt(doc, user_input=False))
    options_input.update(doc2opt(doc, user_input=True))
options.difference_update(('--' + i for i in ('name',) + tqdm.cli.UNSUPPORTED_OPTS))
options_input &= options
options_input -= {'--log'}
completion = u'#!/usr/bin/env bash\n_tqdm(){{\n  local cur prv\n  cur="${{COMP_WORDS[COMP_CWORD]}}"\n  prv="${{COMP_WORDS[COMP_CWORD - 1]}}"\n\n  case ${{prv}} in\n  {opts_manual})\n    # await user input\n    ;;\n  "--log")\n    COMPREPLY=($(compgen -W       \'CRITICAL FATAL ERROR WARN WARNING INFO DEBUG NOTSET\' -- ${{cur}}))\n    ;;\n  *)\n    COMPREPLY=($(compgen -W \'{opts}\' -- ${{cur}}))\n    ;;\n  esac\n}}\ncomplete -F _tqdm tqdm\n'.format(opts=' '.join(sorted(options)), opts_manual='|'.join(sorted(options_input)))
if __name__ == '__main__':
    (Path(__file__).resolve().parent.parent / 'tqdm' / 'completion.sh').write_text(completion, encoding='utf-8')