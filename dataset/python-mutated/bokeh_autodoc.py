""" Integrate Bokeh extensions into Sphinx autodoc.

Ensures that autodoc directives such as ``autoclass`` automatically make use of
Bokeh-specific directives when appropriate. The following Bokeh extensions are
configured:

* :ref:`bokeh.sphinxext.bokeh_color`
* :ref:`bokeh.sphinxext.bokeh_enum`
* :ref:`bokeh.sphinxext.bokeh_model`
* :ref:`bokeh.sphinxext.bokeh_prop`

To enable this extension, add `"bokeh.sphinxext.bokeh_autodoc"` to the
extensions list in your Sphinx configuration module.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from sphinx.ext.autodoc import AttributeDocumenter, ClassDocumenter, ModuleLevelDocumenter
from bokeh.colors.color import Color
from bokeh.core.enums import Enumeration
from bokeh.core.property.descriptors import PropertyDescriptor
from bokeh.model import Model
from . import PARALLEL_SAFE
__all__ = ('ColorDocumenter', 'EnumDocumenter', 'ModelDocumenter', 'PropDocumenter', 'setup')

class ColorDocumenter(ModuleLevelDocumenter):
    directivetype = 'bokeh-color'
    objtype = ''
    priority = 20

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        if False:
            i = 10
            return i + 15
        return isinstance(member, Color)

    def add_content(self, more_content, no_docstring=False):
        if False:
            return 10
        pass

    def get_object_members(self, want_all):
        if False:
            while True:
                i = 10
        return (False, [])

class EnumDocumenter(ModuleLevelDocumenter):
    directivetype = 'bokeh-enum'
    objtype = 'enum'
    priority = 20

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        if False:
            i = 10
            return i + 15
        return isinstance(member, Enumeration)

    def get_object_members(self, want_all):
        if False:
            while True:
                i = 10
        return (False, [])

class PropDocumenter(AttributeDocumenter):
    directivetype = 'bokeh-prop'
    objtype = 'prop'
    priority = 20
    member_order = -100

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(member, PropertyDescriptor)

class ModelDocumenter(ClassDocumenter):
    directivetype = 'bokeh-model'
    objtype = 'model'
    priority = 20

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(member, type) and issubclass(member, Model)

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    ' Required Sphinx extension setup function. '
    app.add_autodocumenter(ColorDocumenter)
    app.add_autodocumenter(EnumDocumenter)
    app.add_autodocumenter(PropDocumenter)
    app.add_autodocumenter(ModelDocumenter)
    return PARALLEL_SAFE