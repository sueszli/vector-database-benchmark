import os
import fitz
root = os.path.abspath(f'{__file__}/../..')

def test_2548():
    if False:
        while True:
            i = 10
    'Text extraction should fail because of PDF structure cycle.\n\n    Old MuPDF version did not detect the loop.\n    '
    print(f'test_2548(): fitz.mupdf_version_tuple={fitz.mupdf_version_tuple!r}')
    if fitz.mupdf_version_tuple < (1, 23, 4):
        print(f'test_2548(): Not testing #2548 because infinite hang before mupdf-1.23.4.')
        return
    fitz.TOOLS.mupdf_warnings(reset=True)
    doc = fitz.open(f'{root}/tests/resources/test_2548.pdf')
    e = False
    for page in doc:
        try:
            _ = page.get_text()
        except Exception as ee:
            print(f'test_2548: ee={ee!r}')
            if hasattr(fitz, 'mupdf'):
                expected = "RuntimeError('code=2: cycle in structure tree')"
            else:
                expected = "RuntimeError('cycle in structure tree')"
            assert repr(ee) == expected, f'Expected expected={expected!r} but got repr(ee)={repr(ee)!r}.'
            e = True
    wt = fitz.TOOLS.mupdf_warnings()
    print(f'test_2548(): wt={wt!r}')
    if fitz.mupdf_version_tuple < (1, 24, 0):
        assert e
        assert not wt
    else:
        expected = 'structure tree broken, assume tree is missing: cycle in structure tree'
        expected = 'cycle in structure tree\nstructure tree broken, assume tree is missing\n' * 76
        expected = expected[:-1]
        expected = 'cycle in structure tree\nstructure tree broken, assume tree is missing'
        assert wt == expected, f'expected:\n    {expected!r}\nwt:\n    {wt!r}\n'
        assert not e