import re
from typing import List, Union
import dagster._check as check
import docutils.nodes as nodes
from dagster._annotations import DeprecatedInfo, ExperimentalInfo
from sphinx.util.docutils import SphinxDirective

def inject_object_flag(obj: object, info: Union[DeprecatedInfo, ExperimentalInfo], docstring: List[str]) -> None:
    if False:
        return 10
    if isinstance(info, DeprecatedInfo):
        additional_text = f' {info.additional_warn_text}.' if info.additional_warn_text else ''
        flag_type = 'deprecated'
        message = f'This API will be removed in version {info.breaking_version}.{additional_text}'
    elif isinstance(info, ExperimentalInfo):
        additional_text = f' {info.additional_warn_text}.' if info.additional_warn_text else ''
        flag_type = 'experimental'
        message = f'This API may break in future versions, even between dot releases.{additional_text}'
    else:
        check.failed(f'Unexpected info type {type(info)}')
    for line in reversed([f'.. flag:: {flag_type}', '', f'   {message}', '']):
        docstring.insert(0, line)

def inject_param_flag(lines: List[str], param: str, info: Union[DeprecatedInfo, ExperimentalInfo]):
    if False:
        while True:
            i = 10
    additional_text = f' {info.additional_warn_text}' if info.additional_warn_text else ''
    if isinstance(info, DeprecatedInfo):
        flag = ':inline-flag:`deprecated`'
        message = f'(This parameter will be removed in version {info.breaking_version}.{additional_text})'
    elif isinstance(info, ExperimentalInfo):
        flag = ':inline-flag:`experimental`'
        message = f'(This parameter may break in future versions, even between dot releases.{additional_text})'
    else:
        check.failed(f'Unexpected info type {type(info)}')
    index = next((i for i in range(len(lines)) if re.search(f'^:param {param}', lines[i])), None)
    modified_line = re.sub(f'^:param {param}:', f':param {param}: {flag} {message}', lines[index]) if index is not None else None
    if index is not None and modified_line is not None:
        lines[index] = modified_line
FLAG_ATTRS = ('flag_type', 'message')

def inline_flag_role(_name, _rawtext, text, _lineno, inliner, _options={}, _content=[]):
    if False:
        i = 10
        return i + 15
    flag_node = inline_flag(flag_type=text)
    return ([flag_node], [])

class inline_flag(nodes.Inline, nodes.TextElement):
    local_attributes = FLAG_ATTRS

def visit_inline_flag(self, node: inline_flag):
    if False:
        while True:
            i = 10
    flag_type = node.attributes['flag_type']
    html = f'\n    <span class="flag {flag_type}">\n      <span class="hidden">(</span>\n      {flag_type}\n      <span class="hidden">)</span>\n    </span>\n    '
    self.body.append(html)

class flag(nodes.Element):
    local_attributes = [*nodes.Element.local_attributes, *FLAG_ATTRS]

def visit_flag(self, node: flag):
    if False:
        return 10
    (flag_type, message) = [node.attributes[k] for k in FLAG_ATTRS]
    message = re.sub('`(\\S+?)`', '<cite>\\1</cite>', message)
    html = f'\n    <div class="flag">\n      <p>\n        <span class="flag {flag_type}">\n          <span class="hidden">(</span>\n          {flag_type}\n          <span class="hidden">)</span>\n        </span>\n        {message}\n      </>\n    </div>\n    '
    self.body.append(html)

def depart_flag(self, node: flag):
    if False:
        for i in range(10):
            print('nop')
    ...

class FlagDirective(SphinxDirective):
    required_arguments = 1
    final_argument_whitespace = True
    has_content = True

    def run(self):
        if False:
            print('Hello World!')
        flag_node = flag()
        flag_node['flag_type'] = self.arguments[0]
        flag_node['message'] = ' '.join(self.content)
        return [flag_node]