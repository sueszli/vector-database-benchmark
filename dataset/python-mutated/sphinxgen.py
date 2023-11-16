from pathlib import Path
import re
from typing import Callable, Dict, Union
from hscommon.build import read_changelog_file, filereplace
from sphinx.cmd.build import build_main as sphinx_build
CHANGELOG_FORMAT = '\n{version} ({date})\n----------------------\n\n{description}\n'

def tixgen(tixurl: str) -> Callable[[str], str]:
    if False:
        return 10
    'This is a filter *generator*. tixurl is a url pattern for the tix with a {0} placeholder\n    for the tix #\n    '
    urlpattern = tixurl.format('\\1')
    R = re.compile('#(\\d+)')
    repl = f'`#\\1 <{urlpattern}>`__'
    return lambda text: R.sub(repl, text)

def gen(basepath: Path, destpath: Path, changelogpath: Path, tixurl: str, confrepl: Union[Dict[str, str], None]=None, confpath: Union[Path, None]=None, changelogtmpl: Union[Path, None]=None) -> None:
    if False:
        return 10
    'Generate sphinx docs with all bells and whistles.\n\n    basepath: The base sphinx source path.\n    destpath: The final path of html files\n    changelogpath: The path to the changelog file to insert in changelog.rst.\n    tixurl: The URL (with one formattable argument for the tix number) to the ticket system.\n    confrepl: Dictionary containing replacements that have to be made in conf.py. {name: replacement}\n    '
    if confrepl is None:
        confrepl = {}
    if confpath is None:
        confpath = Path(basepath, 'conf.tmpl')
    if changelogtmpl is None:
        changelogtmpl = Path(basepath, 'changelog.tmpl')
    changelog = read_changelog_file(changelogpath)
    tix = tixgen(tixurl)
    rendered_logs = []
    for log in changelog:
        description = tix(log['description'])
        description = re.sub('\\[(.*?)\\]\\((.*?)\\)', '`\\1 <\\2>`__', description)
        rendered = CHANGELOG_FORMAT.format(version=log['version'], date=log['date_str'], description=description)
        rendered_logs.append(rendered)
    confrepl['version'] = changelog[0]['version']
    changelog_out = Path(basepath, 'changelog.rst')
    filereplace(changelogtmpl, changelog_out, changelog='\n'.join(rendered_logs))
    if Path(confpath).exists():
        conf_out = Path(basepath, 'conf.py')
        filereplace(confpath, conf_out, **confrepl)
    try:
        sphinx_build([str(basepath), str(destpath)])
    except SystemExit:
        print("Sphinx called sys.exit(), but we're cancelling it because we don't actually want to exit")