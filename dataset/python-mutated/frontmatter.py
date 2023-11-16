__all__ = ['FrontmatterProc']
from .imports import *
from .process import *
from .doclinks import _nbpath2html
from execnb.nbio import *
from fastcore.imports import *
import yaml
_RE_FM_BASE = '^---\\s*\n(.*?\\S+.*?)\n---\\s*'
_re_fm_nb = re.compile(_RE_FM_BASE + '$', flags=re.DOTALL)
_re_fm_md = re.compile(_RE_FM_BASE, flags=re.DOTALL)

def _fm2dict(s: str, nb=True):
    if False:
        print('Hello World!')
    'Load YAML frontmatter into a `dict`'
    re_fm = _re_fm_nb if nb else _re_fm_md
    match = re_fm.search(s.strip())
    return yaml.safe_load(match.group(1)) if match else {}

def _md2dict(s: str):
    if False:
        print('Hello World!')
    'Convert H1 formatted markdown cell to frontmatter dict'
    if '#' not in s:
        return {}
    m = re.search('^#\\s+(\\S.*?)\\s*$', s, flags=re.MULTILINE)
    if not m:
        return {}
    res = {'title': m.group(1)}
    m = re.search('^>\\s+(\\S.*?)\\s*$', s, flags=re.MULTILINE)
    if m:
        res['description'] = m.group(1)
    r = re.findall('^-\\s+(\\S.*:.*\\S)\\s*$', s, flags=re.MULTILINE)
    if r:
        try:
            res.update(yaml.safe_load('\n'.join(r)))
        except Exception as e:
            warn(f'Failed to create YAML dict for:\n{r}\n\n{e}\n')
    return res

def _dict2fm(d):
    if False:
        for i in range(10):
            print('nop')
    return f'---\n{yaml.dump(d)}\n---\n\n'

def _insertfm(nb, fm):
    if False:
        i = 10
        return i + 15
    nb.cells.insert(0, mk_cell(_dict2fm(fm), 'raw'))

class FrontmatterProc(Processor):
    """A YAML and formatted-markdown frontmatter processor"""

    def begin(self):
        if False:
            i = 10
            return i + 15
        self.fm = getattr(self.nb, 'frontmatter_', {})

    def _update(self, f, cell):
        if False:
            for i in range(10):
                print('nop')
        s = cell.get('source')
        if not s:
            return
        d = f(s)
        if not d:
            return
        self.fm.update(d)
        cell.source = None

    def cell(self, cell):
        if False:
            print('Hello World!')
        if cell.cell_type == 'raw':
            self._update(_fm2dict, cell)
        elif cell.cell_type == 'markdown' and 'title' not in self.fm:
            self._update(_md2dict, cell)

    def end(self):
        if False:
            while True:
                i = 10
        self.nb.frontmatter_ = self.fm
        if not self.fm:
            return
        if not hasattr(self.nb, 'path_'):
            raise AttributeError('Notebook missing `path_` attribute.\n\nPlease remove any nbdev-related notebook filters from your _quarto.yml file (e.g. `ipynb-filter: [nbdev_filter]`), since they are no longer supported as of nbdev v2.3. See the v2.3 launch post for more information: https://forums.fast.ai/t/upcoming-changes-in-v2-3-edit-now-released/98905.')
        self.fm.update({'output-file': _nbpath2html(Path(self.nb.path_)).name})
        _insertfm(self.nb, self.fm)