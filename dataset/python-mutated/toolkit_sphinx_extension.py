"""A Sphinx extension to automatically document CKAN's crazy plugins toolkit,
autodoc-style.

Sphinx's autodoc extension can document modules or classes, but although it
masquerades as a module CKAN's plugins toolkit is actually neither a module nor
a class, it's an object-instance of a class, and it's an object with weird
__getattr__ behavior too. Autodoc can't handle it, so we have this custom
Sphinx extension to automate documenting it instead.

This extension plugs into the reading phase of the Sphinx build. It intercepts
the 'toolkit' document (extensions/plugins-toolkit.rst) after Sphinx has read
the reStructuredText source from file. It modifies the source, adding in Sphinx
directives for everything in the plugins toolkit, and then the Sphinx build
continues as normal (just as if the generated reStructuredText had been entered
into plugins-toolkit.rst manually before running Sphinx).

"""
import inspect
from typing import Any, Callable, Optional
import ckan.plugins.toolkit as toolkit

def setup(app: Any):
    if False:
        for i in range(10):
            print('nop')
    'Setup this Sphinx extension. Called once when initializing Sphinx.\n\n    '
    app.connect('source-read', source_read)

def format_function(name: str, function: Callable[..., Any], docstring: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    "Return a Sphinx .. function:: directive for the given function.\n\n    The directive includes the function's docstring if it has one.\n\n    :param name: the name to give to the function in the directive,\n        eg. 'get_converter'\n    :type name: string\n\n    :param function: the function itself\n    :type function: function\n\n    :param docstring: if given, use this instead of introspecting the function\n        to find its actual docstring\n    :type docstring: string\n\n    :returns: a Sphinx .. function:: directive for the function\n    :rtype: string\n\n    "
    template = '.. py:function:: ckan.plugins.toolkit.{function}{args}\n\n{docstring}\n\n'
    argstring = str(inspect.signature(function))
    docstring = docstring or inspect.getdoc(function)
    if docstring is None:
        docstring = ''
    else:
        docstring = '\n'.join(['   ' + line for line in docstring.split('\n')])
    return template.format(function=name, args=argstring, docstring=docstring)

def format_class(name: str, class_: Any, docstring: Optional[str]=None) -> str:
    if False:
        return 10
    "Return a Sphinx .. class:: directive for the given class.\n\n    The directive includes the class's docstring if it has one.\n\n    :param name: the name to give to the class in the directive,\n        eg. 'DefaultDatasetForm'\n    :type name: string\n\n    :param class_: the class itself\n    :type class_: class\n\n    :param docstring: if given, use this instead of introspecting the class\n        to find its actual docstring\n    :type docstring: string\n\n    :returns: a Sphinx .. class:: directive for the class\n    :rtype: string\n\n    "
    template = '.. py:class:: ckan.plugins.toolkit.{cls}\n\n{docstring}\n\n'
    docstring = docstring or inspect.getdoc(class_)
    if docstring is None:
        docstring = ''
    else:
        docstring = '\n'.join(['   ' + line for line in docstring.split('\n')])
    return template.format(cls=name, docstring=docstring)

def format_object(name: str, object_: Any, docstring: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    "Return a Sphinx .. attribute:: directive for the given object.\n\n    The directive includes the object's class's docstring if it has one.\n\n    :param name: the name to give to the object in the directive,\n        eg. 'request'\n    :type name: string\n\n    :param object_: the object itself\n    :type object_: object\n\n    :param docstring: if given, use this instead of introspecting the object\n        to find its actual docstring\n    :type docstring: string\n\n    :returns: a Sphinx .. attribute:: directive for the object\n    :rtype: string\n\n    "
    template = '.. py:attribute:: ckan.plugins.toolkit.{obj}\n\n{docstring}\n\n'
    docstring = docstring or inspect.getdoc(object_)
    if docstring is None:
        docstring = ''
    else:
        docstring = '\n'.join(['   ' + line for line in docstring.split('\n')])
    return template.format(obj=name, docstring=docstring)

def source_read(app: Any, docname: str, source: Any) -> None:
    if False:
        print('Hello World!')
    'Transform the contents of plugins-toolkit.rst to contain reference docs.\n\n    '
    if docname != 'extensions/plugins-toolkit':
        return
    source_ = '\n'
    for (name, thing) in inspect.getmembers(toolkit):
        if name not in toolkit.__all__:
            continue
        custom_docstring = toolkit.docstring_overrides.get(name)
        if inspect.isfunction(thing):
            source_ += format_function(name, thing, docstring=custom_docstring)
        elif inspect.ismethod(thing):
            source_ += format_function(name, thing, docstring=custom_docstring)
        elif inspect.isclass(thing):
            source_ += format_class(name, thing, docstring=custom_docstring)
        elif isinstance(thing, object):
            source_ += format_object(name, thing, docstring=custom_docstring)
        else:
            assert False, "Someone added {name}:{thing} to the plugins toolkit and this Sphinx extension doesn't know how to document that yet. If you're that someone, you need to add a new format_*() function for it here or the docs won't build.".format(name=name, thing=thing)
    source[0] += source_