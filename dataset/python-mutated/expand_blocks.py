import re
import textwrap
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
_code_tab_regex = re.compile('^( *)((`{3,})[^ ].*) tab="(.+)"\\n([\\s\\S]+?)\\n\\1\\3$', re.MULTILINE)
_config_example_regex = re.compile('^( *)((`{3,})toml\\b.*) config-example\\n([\\s\\S]+?)\\n\\1\\3$', re.MULTILINE)

def _code_tab_replace(m):
    if False:
        for i in range(10):
            print('nop')
    (indent, fence_start, fence_end, title, content) = m.groups()
    return f"""{indent}=== ":octicons-file-code-16: {title}"\n{indent}    {fence_start}\n{textwrap.indent(content, '    ')}\n{indent}    {fence_end}\n"""

def _config_example_replace(m):
    if False:
        while True:
            i = 10
    (indent, fence_start, fence_end, content) = m.groups()
    content_without = re.sub(' *\\[tool.hatch\\]\\n', '', content.replace('[tool.hatch.', '['))
    return f"""{indent}=== ":octicons-file-code-16: pyproject.toml"\n{indent}    {fence_start}\n{textwrap.indent(content, '    ')}\n{indent}    {fence_end}\n\n{indent}=== ":octicons-file-code-16: hatch.toml"\n{indent}    {fence_start}\n{textwrap.indent(content_without, '    ')}\n{indent}    {fence_end}\n"""

def on_config(config, **kwargs):
    if False:
        print('Hello World!')
    config.markdown_extensions.append(MyExtension())

class MyExtension(Extension):

    def extendMarkdown(self, md):
        if False:
            for i in range(10):
                print('nop')
        md.preprocessors.register(MyPreprocessor(), 'mypreprocessor', 100)

class MyPreprocessor(Preprocessor):

    def run(self, lines):
        if False:
            print('Hello World!')
        markdown = '\n'.join(lines)
        markdown = _config_example_regex.sub(_config_example_replace, markdown)
        markdown = _code_tab_regex.sub(_code_tab_replace, markdown)
        return markdown.split('\n')