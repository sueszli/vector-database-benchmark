from __future__ import annotations
import functools
import importlib.metadata
import subprocess
import sys
import pytest
XFAIL = ('sentry.sentry_metrics.client.snuba', 'sentry.web.debug_urls')
EXCLUDED = ('sentry.testutils.', 'sentry.web.frontend.debug.')

def extract_packages(text_content: str) -> set[str]:
    if False:
        i = 10
        return i + 15
    return {line.split('==')[0] for line in text_content.splitlines() if '==' in line}

def package_top_level(package: str) -> list[str]:
    if False:
        print('Hello World!')
    dist = importlib.metadata.distribution(package)
    top_level = dist.read_text('top_level.txt')
    if top_level:
        return top_level.split()
    else:
        return []

@functools.lru_cache
def dev_dependencies() -> tuple[str, ...]:
    if False:
        i = 10
        return i + 15
    with open('requirements-dev-frozen.txt') as f:
        dev_packages = extract_packages(f.read())
    with open('requirements-frozen.txt') as f:
        prod_packages = extract_packages(f.read())
    module_names = []
    for package in dev_packages - prod_packages:
        module_names.extend(package_top_level(package))
    return tuple(sorted(module_names))

def validate_package(package: str, excluded: tuple[str, ...], xfail: tuple[str, ...]) -> None:
    if False:
        print('Hello World!')
    script = f"import builtins\nimport sys\n\nDISALLOWED = frozenset({dev_dependencies()!r})\nEXCLUDED = {excluded!r}\nXFAIL = frozenset({xfail!r})\n\norig = builtins.__import__\n\ndef _import(name, globals=None, locals=None, fromlist=(), level=0):\n    base, *_ = name.split('.')\n    if level == 0 and base in DISALLOWED:\n        raise ImportError(f'disallowed dev import: {{name}}')\n    else:\n        return orig(name, globals=globals, locals=locals, fromlist=fromlist, level=level)\n\nbuiltins.__import__ = _import\n\nimport sentry.conf.server_mypy\n\nfrom django.conf import settings\nsettings.DEBUG = False\n\nimport pkgutil\n\npkg = __import__({package!r})\nnames = [\n    name\n    for _, name, _ in pkgutil.walk_packages(pkg.__path__, f'{{pkg.__name__}}.')\n    if name not in XFAIL and not name.startswith(EXCLUDED)\n]\n\nfor name in names:\n    try:\n        __import__(name)\n    except SystemExit:\n        raise SystemExit(f'unexpected exit from {{name}}')\n    except Exception:\n        print(f'error importing {{name}}:', flush=True)\n        print(flush=True)\n        raise\n\nfor xfail in {xfail!r}:\n    try:\n        __import__(xfail)\n    except ImportError:  # expected failure\n        pass\n    else:\n        raise SystemExit(f'unexpected success importing {{xfail}}')\n"
    env = {'SENTRY_ENVIRONMENT': 'production'}
    ret = subprocess.run((sys.executable, '-c', script), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if ret.returncode:
        raise AssertionError(ret.stdout)

@pytest.mark.parametrize('pkg', ('sentry', 'sentry_plugins'))
def test_startup_imports(pkg):
    if False:
        i = 10
        return i + 15
    validate_package(pkg, EXCLUDED, XFAIL)