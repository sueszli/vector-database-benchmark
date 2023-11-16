""" Jinja folklore wrappers and handling of inline copy usage.

"""
import sys
from nuitka.__past__ import unicode
from .Importing import importFromInlineCopy
environments = {}

def unlikely_if(value):
    if False:
        print('Hello World!')
    if value:
        return 'unlikely'
    else:
        return ''

def unlikely_or_likely_from(value):
    if False:
        while True:
            i = 10
    if value:
        return 'unlikely'
    else:
        return 'likely'
_jinja2 = None
_markupsafe = None
_loaded_pkg_resources = None

def getJinja2Package():
    if False:
        for i in range(10):
            print('nop')
    global _jinja2, _markupsafe, _loaded_pkg_resources
    if _markupsafe is None:
        _markupsafe = importFromInlineCopy('markupsafe', must_exist=True)
    if 'pkg_resources' not in sys.modules:
        if _loaded_pkg_resources is None:
            _loaded_pkg_resources = importFromInlineCopy('pkg_resources', must_exist=False)
    if _jinja2 is None:
        _jinja2 = importFromInlineCopy('jinja2', must_exist=True)
    if _loaded_pkg_resources is not None and 'pkg_resources' in sys.modules:
        del sys.modules['pkg_resources']
    return _jinja2

def getEnvironment(package_name, template_subdir, extensions):
    if False:
        while True:
            i = 10
    key = (package_name, template_subdir, extensions)
    if key not in environments:
        jinja2 = getJinja2Package()
        if package_name is not None:
            loader = jinja2.PackageLoader(package_name, template_subdir)
        elif template_subdir is not None:
            loader = jinja2.FileSystemLoader(template_subdir)
        else:
            loader = jinja2.BaseLoader()
        env = jinja2.Environment(loader=loader, extensions=extensions, trim_blocks=True, lstrip_blocks=True)
        env.globals.update({'unlikely_if': unlikely_if, 'unlikely_or_likely_from': unlikely_or_likely_from})
        env.undefined = jinja2.StrictUndefined
        environments[key] = env
    return environments[key]

def getTemplate(package_name, template_name, template_subdir='templates', extensions=()):
    if False:
        i = 10
        return i + 15
    return getEnvironment(package_name=package_name, template_subdir=template_subdir, extensions=extensions).get_template(template_name)

def getTemplateC(package_name, template_name, template_subdir='templates_c', extensions=()):
    if False:
        for i in range(10):
            print('nop')
    return getEnvironment(package_name=package_name, template_subdir=template_subdir, extensions=extensions).get_template(template_name)

def getTemplateFromString(template_str):
    if False:
        return 10
    return getEnvironment(package_name=None, template_subdir=None, extensions=()).from_string(template_str.strip())
_template_cache = {}

def renderTemplateFromString(template_str, **kwargs):
    if False:
        i = 10
        return i + 15
    if template_str not in _template_cache:
        _template_cache[template_str] = getTemplateFromString(template_str)
    result = _template_cache[template_str].render(**kwargs)
    if str is not unicode:
        return result.encode('utf8')
    else:
        return result