"""
Verify the reproducibility of gettext machine objects (.mo) from catalogs
(.po).

Due to tool- and library-level idiosyncrasies, this happens in three stages:

1. Via polib: Overwrite metadata .mo → .po.
2. Via translate: Recompile the entire catalog .po → .mo.
3. Via diffoscope: Diff the new .mo against the old, heavily masked and
   filtered to avoid false positives from stray entries in the "fuzzy"
   and "obsolete" states.

In other words, the new .mo file should be identical (modulo stray entries) to
the original, meaning that the original .po/.mo pair differed only in their
metadata.
"""
import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterator, Optional, Set
import polib
from translate.tools.pocompile import convertmo
parser = argparse.ArgumentParser('Verify the reproducibility of gettext machine objects (.mo) from catalogs (.po).')
parser.add_argument('locale', nargs='+', help='one or more locale directories, each of which must contain an\n    "LC_MESSAGES" directory')
parser.add_argument('--domain', default='messages', help='the gettext domain to load (defaults to "messages")')
args = parser.parse_args()

class CatalogVerifier:
    """Wrapper class for proving .mo → .po → .mo reproducibility."""

    def __init__(self, path: Path, domain: str):
        if False:
            i = 10
            return i + 15
        'Set up the .po/.mo pair.'
        self.path = path
        self.po = polib.pofile(str(path / 'LC_MESSAGES' / f'{domain}.po'))
        self.mo = polib.mofile(str(path / 'LC_MESSAGES' / f'{domain}.mo'))

    def __enter__(self) -> 'CatalogVerifier':
        if False:
            print('Hello World!')
        'Prepare to generate the new .mo file to diff.'
        self.mo_target = Path(f'{self.mo.fpath}.new')
        return self

    def __exit__(self, exc_type: Optional[Any], exc_value: Optional[Any], traceback: Optional[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clean up.'
        self.mo_target.unlink(missing_ok=True)

    @property
    def strays(self) -> Set[str]:
        if False:
            return 10
        'Return the set of stray (fuzzy or obsolete) entries to mask when\n        diffing this catalog.'
        fuzzy = {f"^{line.replace('#| ', '')}" for e in self.po.fuzzy_entries() for line in str(e).splitlines()}
        obsolete = {f"^{line.replace('#~ ', '')}" for e in self.po.obsolete_entries() for line in str(e).splitlines()}
        return fuzzy | obsolete

    def diffoscope_args(self, a: Path, b: Path, filtered: bool=True) -> Iterator[str]:
        if False:
            while True:
                i = 10
        'Build up a diffoscope invocation that (with `filtered`) removes\n        false positives from the msgunfmt diff.'
        yield f'diffoscope {a} {b}'
        if not filtered:
            return
        yield "--diff-mask '^$'"
        for stray in self.strays:
            yield f'--diff-mask {shlex.quote(stray)}'
        yield "| grep -Fv '[masked]'"
        yield "| grep -E '│ (-|\\+)msg(id|str)'"

    def diffoscope_call(self, a: Path, b: Path, filtered: bool=True) -> subprocess.CompletedProcess:
        if False:
            return 10
        'Call diffoscope and return the subprocess.CompletedProcess result\n        for further processing, *without* first checking whether it was\n        succesful.'
        cmd = ' '.join(self.diffoscope_args(a, b, filtered))
        return subprocess.run(cmd, capture_output=True, env=os.environ, shell=True)

    def reproduce(self) -> None:
        if False:
            return 10
        'Overwrite metadata .mo → .po.  Then rewrite the entire file .po →\n        .mo.'
        self.po.metadata = self.mo.metadata
        self.po.save(self.po.fpath)
        with open(self.mo_target, 'wb') as mo_target:
            convertmo(self.po.fpath, mo_target, '')

    def verify(self) -> None:
        if False:
            i = 10
            return i + 15
        "Run diffoscope for this catalog and error if there's any unmasked\n        diff."
        test = self.diffoscope_call(Path(self.mo.fpath), Path(self.mo_target), filtered=False)
        if test.returncode not in [0, 1]:
            test.check_returncode()
        result = self.diffoscope_call(Path(self.mo.fpath), Path(self.mo_target))
        print(f'--> Verifying {self.path}: {result.args}')
        if len(result.stdout) > 0:
            raise Exception(result.stdout.decode('utf-8'))
print(f'--> Reproducing {len(args.locale)} path(s)')
for path in args.locale:
    locale_dir = Path(path).resolve()
    if not locale_dir.is_dir():
        print(f'--> Skipping "{locale_dir}"')
        continue
    with CatalogVerifier(locale_dir, args.domain) as catalog:
        catalog.reproduce()
        catalog.verify()