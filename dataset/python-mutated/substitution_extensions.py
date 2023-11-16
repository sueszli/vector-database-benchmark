from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any
from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.roles import code_role
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms.code import HighlightLanguageTransform
if TYPE_CHECKING:
    from docutils.nodes import Node, system_message
    from sphinx.application import Sphinx
LOGGER = logging.getLogger(__name__)
OriginalCodeBlock: Directive = directives._directives['code-block']
_SUBSTITUTION_OPTION_NAME = 'substitutions'

class SubstitutionCodeBlock(OriginalCodeBlock):
    """Similar to CodeBlock but replaces placeholders with variables."""
    option_spec = OriginalCodeBlock.option_spec.copy()
    option_spec[_SUBSTITUTION_OPTION_NAME] = directives.flag

    def run(self) -> list:
        if False:
            while True:
                i = 10
        'Decorate code block so that SubstitutionCodeBlockTransform will notice it'
        [node] = super().run()
        if _SUBSTITUTION_OPTION_NAME in self.options:
            node.attributes['substitutions'] = True
        return [node]

class SubstitutionCodeBlockTransform(SphinxTransform):
    """Substitute ``|variables|`` in code and code-block nodes"""
    default_priority = HighlightLanguageTransform.default_priority - 1

    def apply(self, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15

        def condition(node):
            if False:
                for i in range(10):
                    print('nop')
            return isinstance(node, (nodes.literal_block, nodes.literal))
        for node in self.document.traverse(condition):
            if _SUBSTITUTION_OPTION_NAME not in node:
                continue
            document = node.document
            parent = node.parent
            while document is None:
                parent = parent.parent
                document = parent.document
            substitution_defs = document.substitution_defs
            for child in node.children:
                old_child = child
                for (name, value) in substitution_defs.items():
                    replacement = value.astext()
                    child = nodes.Text(child.replace(f'|{name}|', replacement))
                node.replace(old_child, child)
            node.rawsource = node.astext()

def substitution_code_role(*args, **kwargs) -> tuple[list[Node], list[system_message]]:
    if False:
        return 10
    'Decorate an inline code so that SubstitutionCodeBlockTransform will notice it'
    ([node], system_messages) = code_role(*args, **kwargs)
    node[_SUBSTITUTION_OPTION_NAME] = True
    return ([node], system_messages)
substitution_code_role.options = {'class': directives.class_option, 'language': directives.unchanged}

class AddSpacepadSubstReference(SphinxTransform):
    """
    Add a custom ``|version-spacepad|`` replacement definition

    Since this desired replacement text is all just whitespace, we can't use
    the normal RST to define this, we instead of to create this definition
    manually after docutils has parsed the source files.
    """
    default_priority = 1

    def apply(self, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        substitution_defs = self.document.substitution_defs
        version = substitution_defs['version'].astext()
        pad = ' ' * len(version)
        substitution_defs['version-spacepad'] = nodes.substitution_definition(version, pad)
        ...

def setup(app: Sphinx) -> dict:
    if False:
        for i in range(10):
            print('nop')
    'Setup plugin'
    app.add_config_value('substitutions', [], 'html')
    directives.register_directive('code-block', SubstitutionCodeBlock)
    app.add_role('subst-code', substitution_code_role)
    app.add_post_transform(SubstitutionCodeBlockTransform)
    app.add_post_transform(AddSpacepadSubstReference)
    return {'parallel_write_safe': True, 'parallel_read_safe': True}