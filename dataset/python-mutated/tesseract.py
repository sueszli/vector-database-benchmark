"""Interface to Tesseract executable."""
from __future__ import annotations
import logging
import re
from contextlib import suppress
from math import pi
from os import fspath
from pathlib import Path
from subprocess import PIPE, STDOUT, CalledProcessError, TimeoutExpired
from packaging.version import Version
from ocrmypdf.exceptions import MissingDependencyError, SubprocessOutputError, TesseractConfigError
from ocrmypdf.pluginspec import OrientationConfidence
from ocrmypdf.subprocess import get_version, run
log = logging.getLogger(__name__)
TESSERACT_THRESHOLDING_METHODS: dict[str, int] = {'auto': 0, 'otsu': 0, 'adaptive-otsu': 1, 'sauvola': 2}

class TesseractLoggerAdapter(logging.LoggerAdapter):
    """Prepend [tesseract] to messages emitted from tesseract."""

    def process(self, msg, kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs['extra'] = self.extra
        return (f'[tesseract] {msg}', kwargs)
TESSERACT_VERSION_PATTERN = '\n    v?\n    (?:\n        (?:(?P<epoch>[0-9]+)!)?                           # epoch\n        (?P<release>[0-9]+(?:\\.[0-9]+)*)                  # release segment\n        (?P<pre>                                          # pre-release\n            [-_\\.]?\n            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))\n            [-_\\.]?\n            (?P<pre_n>[0-9]+)?\n        )?\n        (?P<post>                                         # post release\n            (?:-(?P<post_n1>[0-9]+))\n            |\n            (?:\n                [-_\\.]?\n                (?P<post_l>post|rev|r)\n                [-_\\.]?\n                (?P<post_n2>[0-9]+)?\n            )\n        )?\n        (?P<dev>                                          # dev release\n            [-_\\.]?\n            (?P<dev_l>dev)\n            [-_\\.]?\n            (?P<dev_n>[0-9]+)?\n        )?\n        (?P<date>\n            [-_\\.]\n            (?:20[0-9][0-9] [0-1][0-9] [0-3][0-9])       # yyyy mm dd\n        )?\n        (?P<gitcount>\n            [-_\\.]?\n            [0-9]+\n        )?\n        (?P<gitcommit>\n            [-_\\.]?\n            g[0-9a-f]{2,10}\n        )?\n    )\n    (?:\\+(?P<local>[a-z0-9]+(?:[-_\\.][a-z0-9]+)*))?       # local version\n'

class TesseractVersion(Version):
    """Modify standard packaging.Version regex to support Tesseract idiosyncrasies."""
    _regex = re.compile('^\\s*' + TESSERACT_VERSION_PATTERN + '\\s*$', re.VERBOSE | re.IGNORECASE)

def version() -> Version:
    if False:
        for i in range(10):
            print('nop')
    return TesseractVersion(get_version('tesseract', regex='tesseract\\s(.+)'))

def has_thresholding() -> bool:
    if False:
        while True:
            i = 10
    'Does Tesseract have -c thresholding method capability?'
    return version() >= Version('5.0')

def get_languages() -> set[str]:
    if False:
        for i in range(10):
            print('nop')

    def lang_error(output):
        if False:
            i = 10
            return i + 15
        msg = 'Tesseract failed to report available languages.\nOutput from Tesseract:\n-----------\n'
        msg += output
        return msg
    args_tess = ['tesseract', '--list-langs']
    try:
        proc = run(args_tess, text=True, stdout=PIPE, stderr=STDOUT, logs_errors_to_stdout=True, check=True)
        output = proc.stdout
    except CalledProcessError as e:
        raise MissingDependencyError(lang_error(e.output)) from e
    for line in output.splitlines():
        if line.startswith('Error'):
            raise MissingDependencyError(lang_error(output))
    (_header, *rest) = output.splitlines()
    return {lang.strip() for lang in rest}

def tess_base_args(langs: list[str], engine_mode: int | None) -> list[str]:
    if False:
        print('Hello World!')
    args = ['tesseract']
    if langs:
        args.extend(['-l', '+'.join(langs)])
    if engine_mode is not None:
        args.extend(['--oem', str(engine_mode)])
    return args

def _parse_tesseract_output(binary_output: bytes) -> dict[str, str]:
    if False:
        print('Hello World!')

    def gen():
        if False:
            i = 10
            return i + 15
        for line in binary_output.decode().splitlines():
            line = line.strip()
            parts = line.split(':', maxsplit=2)
            if len(parts) == 2:
                yield (parts[0].strip(), parts[1].strip())
    return dict(gen())

def get_orientation(input_file: Path, engine_mode: int | None, timeout: float) -> OrientationConfidence:
    if False:
        print('Hello World!')
    args_tesseract = tess_base_args(['osd'], engine_mode) + ['--psm', '0', fspath(input_file), 'stdout']
    try:
        p = run(args_tesseract, stdout=PIPE, stderr=STDOUT, timeout=timeout, check=True)
    except TimeoutExpired:
        return OrientationConfidence(angle=0, confidence=0.0)
    except CalledProcessError as e:
        tesseract_log_output(e.stdout)
        tesseract_log_output(e.stderr)
        if b'Too few characters. Skipping this page' in e.output or b'Image too large' in e.output:
            return OrientationConfidence(0, 0)
        raise SubprocessOutputError() from e
    osd = _parse_tesseract_output(p.stdout)
    angle = int(osd.get('Orientation in degrees', 0))
    orient_conf = OrientationConfidence(angle=angle, confidence=float(osd.get('Orientation confidence', 0)))
    return orient_conf

def get_deskew(input_file: Path, languages: list[str], engine_mode: int | None, timeout: float) -> float:
    if False:
        while True:
            i = 10
    'Gets angle to deskew this page, in degrees.'
    args_tesseract = tess_base_args(languages, engine_mode) + ['--psm', '2', fspath(input_file), 'stdout']
    try:
        p = run(args_tesseract, stdout=PIPE, stderr=STDOUT, timeout=timeout, check=True)
    except TimeoutExpired:
        return 0.0
    except CalledProcessError as e:
        tesseract_log_output(e.stdout)
        tesseract_log_output(e.stderr)
        if b'Empty page!!' in e.output or (e.output == b'' and e.returncode == 1):
            return 0.0
        raise SubprocessOutputError() from e
    parsed = _parse_tesseract_output(p.stdout)
    deskew_radians = float(parsed.get('Deskew angle', 0))
    deskew_degrees = 180 / pi * deskew_radians
    log.debug(f'Deskew angle: {deskew_degrees:.3f}')
    return deskew_degrees

def tesseract_log_output(stream: bytes) -> None:
    if False:
        return 10
    tlog = TesseractLoggerAdapter(log, extra=log.extra if hasattr(log, 'extra') else None)
    if not stream:
        return
    try:
        text = stream.decode()
    except UnicodeDecodeError:
        text = stream.decode('utf-8', 'ignore')
    lines = text.splitlines()
    for line in lines:
        if line.startswith('Tesseract Open Source'):
            continue
        elif line.startswith('Warning in pixReadMem'):
            continue
        elif 'diacritics' in line:
            tlog.warning('lots of diacritics - possibly poor OCR')
        elif line.startswith('OSD: Weak margin'):
            tlog.warning('unsure about page orientation')
        elif 'Error in pixScanForForeground' in line:
            pass
        elif 'Error in boxClipToRectangle' in line:
            pass
        elif 'parameter not found: ' in line.lower():
            tlog.error(line.strip())
            problem = line.split('found: ')[1]
            raise TesseractConfigError(problem)
        elif 'error' in line.lower() or 'exception' in line.lower():
            tlog.error(line.strip())
        elif 'warning' in line.lower():
            tlog.warning(line.strip())
        elif 'read_params_file' in line.lower():
            tlog.error(line.strip())
        else:
            tlog.info(line.strip())

def page_timedout(timeout: float) -> None:
    if False:
        print('Hello World!')
    if timeout == 0:
        return
    log.warning('[tesseract] took too long to OCR - skipping')

def _generate_null_hocr(output_hocr: Path, output_text: Path, image: Path) -> None:
    if False:
        return 10
    'Produce an empty .hocr file.\n\n    Ensures page is the same size as the input image.\n    '
    output_hocr.write_text('', encoding='utf-8')
    output_text.write_text('[skipped page]', encoding='utf-8')

def generate_hocr(*, input_file: Path, output_hocr: Path, output_text: Path, languages: list[str], engine_mode: int, tessconfig: list[str], timeout: float, pagesegmode: int, thresholding: int, user_words, user_patterns) -> None:
    if False:
        while True:
            i = 10
    'Generate a hOCR file, which must be converted to PDF.'
    prefix = output_hocr.with_suffix('')
    args_tesseract = tess_base_args(languages, engine_mode)
    if pagesegmode is not None:
        args_tesseract.extend(['--psm', str(pagesegmode)])
    if thresholding != 0 and has_thresholding():
        args_tesseract.extend(['-c', f'thresholding_method={thresholding}'])
    if user_words:
        args_tesseract.extend(['--user-words', user_words])
    if user_patterns:
        args_tesseract.extend(['--user-patterns', user_patterns])
    args_tesseract.extend([fspath(input_file), fspath(prefix), 'hocr', 'txt'])
    args_tesseract.extend(tessconfig)
    try:
        p = run(args_tesseract, stdout=PIPE, stderr=STDOUT, timeout=timeout, check=True)
        stdout = p.stdout
    except TimeoutExpired:
        page_timedout(timeout)
        _generate_null_hocr(output_hocr, output_text, input_file)
    except CalledProcessError as e:
        tesseract_log_output(e.output)
        if b'Image too large' in e.output or b'Empty page!!' in e.output:
            _generate_null_hocr(output_hocr, output_text, input_file)
            return
        raise SubprocessOutputError() from e
    else:
        tesseract_log_output(stdout)
        with suppress(FileNotFoundError):
            prefix.with_suffix('.txt').replace(output_text)

def use_skip_page(output_pdf: Path, output_text: Path) -> None:
    if False:
        return 10
    output_text.write_text('[skipped page]', encoding='utf-8')
    output_pdf.write_bytes(b'')

def generate_pdf(*, input_file: Path, output_pdf: Path, output_text: Path, languages: list[str], engine_mode: int, tessconfig: list[str], timeout: float, pagesegmode: int, thresholding: int, user_words, user_patterns) -> None:
    if False:
        print('Hello World!')
    "Generate a PDF using Tesseract's internal PDF generator.\n\n    We specifically a text-only PDF which is more suitable for combining with\n    the input page.\n    "
    args_tesseract = tess_base_args(languages, engine_mode)
    if pagesegmode is not None:
        args_tesseract.extend(['--psm', str(pagesegmode)])
    args_tesseract.extend(['-c', 'textonly_pdf=1'])
    if thresholding != 0 and has_thresholding():
        args_tesseract.extend(['-c', f'thresholding_method={thresholding}'])
    if user_words:
        args_tesseract.extend(['--user-words', user_words])
    if user_patterns:
        args_tesseract.extend(['--user-patterns', user_patterns])
    prefix = output_pdf.parent / Path(output_pdf.stem)
    args_tesseract.extend([fspath(input_file), fspath(prefix), 'pdf', 'txt'])
    args_tesseract.extend(tessconfig)
    try:
        p = run(args_tesseract, stdout=PIPE, stderr=STDOUT, timeout=timeout, check=True)
        stdout = p.stdout
        with suppress(FileNotFoundError):
            prefix.with_suffix('.txt').replace(output_text)
    except TimeoutExpired:
        page_timedout(timeout)
        use_skip_page(output_pdf, output_text)
    except CalledProcessError as e:
        tesseract_log_output(e.output)
        if b'Image too large' in e.output or b'Empty page!!' in e.output:
            use_skip_page(output_pdf, output_text)
            return
        raise SubprocessOutputError() from e
    else:
        tesseract_log_output(stdout)