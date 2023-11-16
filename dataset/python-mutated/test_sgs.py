from argparse import Namespace
import glob
from pathlib import Path
import tempfile
import pytest
from . import sgs
FIXTURE_INGREDIENTS = Path('sgs_test_fixtures/ingredients')
FIXTURE_RECIPES = Path('sgs_test_fixtures/recipes')
FIXTURE_OUTPUT = Path('sgs_test_fixtures/output')

def test_sgs_generate():
    if False:
        i = 10
        return i + 15
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = Namespace(output_dir=tmp_dir)
        sgs.generate(args, FIXTURE_INGREDIENTS.absolute(), FIXTURE_RECIPES.absolute())
        for test_file in map(Path, glob.glob(f'{tmp_dir}/**')):
            match_file = FIXTURE_OUTPUT / test_file.relative_to(tmp_dir)
            assert test_file.read_bytes() == match_file.read_bytes()

def test_snippets_freshness():
    if False:
        for i in range(10):
            print('nop')
    '\n    Make sure that the snippets/ folder is up-to-date and matches\n    ingredients/ and recipes/. This test will generate SGS output\n    in a temporary directory and compare it to the content of\n    snippets/ folder.\n    '
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = Namespace(output_dir=tmp_dir)
        sgs.generate(args, Path('ingredients/').absolute(), Path('recipes/').absolute())
        print(list(map(Path, glob.glob(f'{tmp_dir}/**'))))
        for test_file in map(Path, glob.glob(f'{tmp_dir}/**', recursive=True)):
            match_file = Path('snippets/') / test_file.relative_to(tmp_dir)
            if test_file.is_file():
                if test_file.read_bytes() != match_file.read_bytes():
                    pytest.fail(f'This test fails because file {match_file} seems to be outdated. Please run `python sgs.py generate` to update your snippets.')
            elif test_file.is_dir():
                assert match_file.is_dir()