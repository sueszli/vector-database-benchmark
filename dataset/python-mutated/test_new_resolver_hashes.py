import collections
import hashlib
import pytest
from tests.lib import PipTestEnvironment, create_basic_sdist_for_package, create_basic_wheel_for_package
_FindLinks = collections.namedtuple('_FindLinks', 'index_html sdist_hash wheel_hash')

def _create_find_links(script: PipTestEnvironment) -> _FindLinks:
    if False:
        i = 10
        return i + 15
    sdist_path = create_basic_sdist_for_package(script, 'base', '0.1.0')
    wheel_path = create_basic_wheel_for_package(script, 'base', '0.1.0')
    sdist_hash = hashlib.sha256(sdist_path.read_bytes()).hexdigest()
    wheel_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    index_html = script.scratch_path / 'index.html'
    index_html.write_text(f'\n        <!DOCTYPE html>\n        <a href="{sdist_path.as_uri()}#sha256={sdist_hash}">{sdist_path.stem}</a>\n        <a href="{wheel_path.as_uri()}#sha256={wheel_hash}">{wheel_path.stem}</a>\n        '.strip())
    return _FindLinks(index_html, sdist_hash, wheel_hash)

@pytest.mark.parametrize('requirements_template, message', [('\n            base==0.1.0 --hash=sha256:{sdist_hash} --hash=sha256:{wheel_hash}\n            base==0.1.0 --hash=sha256:{sdist_hash} --hash=sha256:{wheel_hash}\n            ', 'Checked 2 links for project {name!r} against 2 hashes (2 matches, 0 no digest): discarding no candidates'), ('\n            base==0.1.0 --hash=sha256:{sdist_hash} --hash=sha256:{wheel_hash}\n            base==0.1.0 --hash=sha256:{sdist_hash}\n            ', 'Checked 2 links for project {name!r} against 1 hashes (1 matches, 0 no digest): discarding 1 non-matches')], ids=['identical', 'intersect'])
def test_new_resolver_hash_intersect(script: PipTestEnvironment, requirements_template: str, message: str) -> None:
    if False:
        while True:
            i = 10
    find_links = _create_find_links(script)
    requirements_txt = script.scratch_path / 'requirements.txt'
    requirements_txt.write_text(requirements_template.format(sdist_hash=find_links.sdist_hash, wheel_hash=find_links.wheel_hash))
    result = script.pip('install', '--no-cache-dir', '--no-deps', '--no-index', '--find-links', find_links.index_html, '-vv', '--requirement', requirements_txt)
    assert message.format(name='base') in result.stdout, str(result)

def test_new_resolver_hash_intersect_from_constraint(script: PipTestEnvironment) -> None:
    if False:
        return 10
    find_links = _create_find_links(script)
    constraints_txt = script.scratch_path / 'constraints.txt'
    constraints_txt.write_text(f'base==0.1.0 --hash=sha256:{find_links.sdist_hash}')
    requirements_txt = script.scratch_path / 'requirements.txt'
    requirements_txt.write_text('\n        base==0.1.0 --hash=sha256:{sdist_hash} --hash=sha256:{wheel_hash}\n        '.format(sdist_hash=find_links.sdist_hash, wheel_hash=find_links.wheel_hash))
    result = script.pip('install', '--no-cache-dir', '--no-deps', '--no-index', '--find-links', find_links.index_html, '-vv', '--constraint', constraints_txt, '--requirement', requirements_txt)
    message = 'Checked 2 links for project {name!r} against 1 hashes (1 matches, 0 no digest): discarding 1 non-matches'.format(name='base')
    assert message in result.stdout, str(result)

@pytest.mark.parametrize('requirements_template, constraints_template', [('\n            base==0.1.0 --hash=sha256:{sdist_hash}\n            base==0.1.0 --hash=sha256:{wheel_hash}\n            ', ''), ('base==0.1.0 --hash=sha256:{sdist_hash}', 'base==0.1.0 --hash=sha256:{wheel_hash}')], ids=['both-requirements', 'one-each'])
def test_new_resolver_hash_intersect_empty(script: PipTestEnvironment, requirements_template: str, constraints_template: str) -> None:
    if False:
        i = 10
        return i + 15
    find_links = _create_find_links(script)
    constraints_txt = script.scratch_path / 'constraints.txt'
    constraints_txt.write_text(constraints_template.format(sdist_hash=find_links.sdist_hash, wheel_hash=find_links.wheel_hash))
    requirements_txt = script.scratch_path / 'requirements.txt'
    requirements_txt.write_text(requirements_template.format(sdist_hash=find_links.sdist_hash, wheel_hash=find_links.wheel_hash))
    result = script.pip('install', '--no-cache-dir', '--no-deps', '--no-index', '--find-links', find_links.index_html, '--constraint', constraints_txt, '--requirement', requirements_txt, expect_error=True)
    assert 'THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE.' in result.stderr, str(result)

def test_new_resolver_hash_intersect_empty_from_constraint(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    find_links = _create_find_links(script)
    constraints_txt = script.scratch_path / 'constraints.txt'
    constraints_txt.write_text(f'\n        base==0.1.0 --hash=sha256:{find_links.sdist_hash}\n        base==0.1.0 --hash=sha256:{find_links.wheel_hash}\n        ')
    result = script.pip('install', '--no-cache-dir', '--no-deps', '--no-index', '--find-links', find_links.index_html, '--constraint', constraints_txt, 'base==0.1.0', expect_error=True)
    message = 'Hashes are required in --require-hashes mode, but they are missing from some requirements.'
    assert message in result.stderr, str(result)

@pytest.mark.parametrize('constrain_by_hash', [False, True])
def test_new_resolver_hash_requirement_and_url_constraint_can_succeed(script: PipTestEnvironment, constrain_by_hash: bool) -> None:
    if False:
        i = 10
        return i + 15
    wheel_path = create_basic_wheel_for_package(script, 'base', '0.1.0')
    wheel_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    requirements_txt = script.scratch_path / 'requirements.txt'
    requirements_txt.write_text(f'\n        base==0.1.0 --hash=sha256:{wheel_hash}\n        ')
    constraints_txt = script.scratch_path / 'constraints.txt'
    constraint_text = f'base @ {wheel_path.as_uri()}\n'
    if constrain_by_hash:
        constraint_text += f'base==0.1.0 --hash=sha256:{wheel_hash}\n'
    constraints_txt.write_text(constraint_text)
    script.pip('install', '--no-cache-dir', '--no-index', '--constraint', constraints_txt, '--requirement', requirements_txt)
    script.assert_installed(base='0.1.0')

@pytest.mark.parametrize('constrain_by_hash', [False, True])
def test_new_resolver_hash_requirement_and_url_constraint_can_fail(script: PipTestEnvironment, constrain_by_hash: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    wheel_path = create_basic_wheel_for_package(script, 'base', '0.1.0')
    other_path = create_basic_wheel_for_package(script, 'other', '0.1.0')
    other_hash = hashlib.sha256(other_path.read_bytes()).hexdigest()
    requirements_txt = script.scratch_path / 'requirements.txt'
    requirements_txt.write_text(f'\n        base==0.1.0 --hash=sha256:{other_hash}\n        ')
    constraints_txt = script.scratch_path / 'constraints.txt'
    constraint_text = f'base @ {wheel_path.as_uri()}\n'
    if constrain_by_hash:
        constraint_text += f'base==0.1.0 --hash=sha256:{other_hash}\n'
    constraints_txt.write_text(constraint_text)
    result = script.pip('install', '--no-cache-dir', '--no-index', '--constraint', constraints_txt, '--requirement', requirements_txt, expect_error=True)
    assert 'THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE.' in result.stderr, str(result)
    script.assert_not_installed('base', 'other')

def test_new_resolver_hash_with_extras(script: PipTestEnvironment) -> None:
    if False:
        print('Hello World!')
    parent_with_extra_path = create_basic_wheel_for_package(script, 'parent_with_extra', '0.1.0', depends=['child[extra]'])
    parent_with_extra_hash = hashlib.sha256(parent_with_extra_path.read_bytes()).hexdigest()
    parent_without_extra_path = create_basic_wheel_for_package(script, 'parent_without_extra', '0.1.0', depends=['child'])
    parent_without_extra_hash = hashlib.sha256(parent_without_extra_path.read_bytes()).hexdigest()
    child_path = create_basic_wheel_for_package(script, 'child', '0.1.0', extras={'extra': ['extra']})
    child_hash = hashlib.sha256(child_path.read_bytes()).hexdigest()
    create_basic_wheel_for_package(script, 'child', '0.2.0', extras={'extra': ['extra']})
    extra_path = create_basic_wheel_for_package(script, 'extra', '0.1.0')
    extra_hash = hashlib.sha256(extra_path.read_bytes()).hexdigest()
    requirements_txt = script.scratch_path / 'requirements.txt'
    requirements_txt.write_text(f'\n        child[extra]==0.1.0 --hash=sha256:{child_hash}\n        parent_with_extra==0.1.0 --hash=sha256:{parent_with_extra_hash}\n        parent_without_extra==0.1.0 --hash=sha256:{parent_without_extra_hash}\n        extra==0.1.0 --hash=sha256:{extra_hash}\n        ')
    script.pip('install', '--no-cache-dir', '--no-index', '--find-links', script.scratch_path, '--requirement', requirements_txt)
    script.assert_installed(parent_with_extra='0.1.0', parent_without_extra='0.1.0', child='0.1.0', extra='0.1.0')