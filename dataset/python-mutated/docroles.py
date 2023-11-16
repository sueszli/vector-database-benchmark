"""Document roles"""
from __future__ import annotations
from functools import partial
from docutils import nodes, utils
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock

class RoleException(Exception):
    """Exception for roles extension"""

def get_template_field(env, fullname) -> list[str]:
    if False:
        i = 10
        return i + 15
    '\n    Gets template fields for specific operator class.\n\n    :param env: env config\n    :param fullname: Full path to operator class.\n        For example: ``airflow.providers.google.cloud.operators.vision.CloudVisionCreateProductSetOperator``\n    :return: List of template field\n    '
    (modname, classname) = fullname.rsplit('.', 1)
    try:
        with mock(env.config.autodoc_mock_imports):
            mod = import_module(modname)
    except ImportError:
        raise RoleException(f'Error loading {modname} module.')
    clazz = getattr(mod, classname)
    if not clazz:
        raise RoleException(f'Error finding {classname} class in {modname} module.')
    template_fields = getattr(clazz, 'template_fields')
    if not template_fields:
        raise RoleException(f'Could not find the template fields for {classname} class in {modname} module.')
    return list(template_fields)

def template_field_role(app, typ, rawtext, text, lineno, inliner, options=None, content=None):
    if False:
        while True:
            i = 10
    '\n    A role that allows you to include a list of template fields in the middle of the text. This is especially\n    useful when writing guides describing how to use the operator.\n    The result is a list of fields where each field is shorted in the literal block.\n\n    Sample usage::\n\n    :template-fields:`airflow.operators.bash.BashOperator`\n\n    For further information look at:\n\n    * [http://docutils.sourceforge.net/docs/howto/rst-roles.html](Creating reStructuredText Interpreted\n      Text Roles)\n    '
    if options is None:
        options = {}
    if content is None:
        content = []
    text = utils.unescape(text)
    try:
        template_fields = get_template_field(app.env, text)
    except RoleException as e:
        msg = inliner.reporter.error(f'invalid class name {text} \n{e}', line=lineno)
        prb = inliner.problematic(rawtext, rawtext, msg)
        return ([prb], [msg])
    node = nodes.inline(rawtext=rawtext)
    for (i, field) in enumerate(template_fields):
        if i != 0:
            node += nodes.Text(', ')
        node += nodes.literal(field, '', nodes.Text(field))
    return ([node], [])

def setup(app):
    if False:
        return 10
    'Sets the extension up'
    from docutils.parsers.rst import roles
    roles.register_local_role('template-fields', partial(template_field_role, app))
    return {'version': 'builtin', 'parallel_read_safe': True, 'parallel_write_safe': True}