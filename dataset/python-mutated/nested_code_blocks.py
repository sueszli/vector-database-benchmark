from typing import Any, List, Mapping, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement
import markdown
from markdown.extensions import Extension
from typing_extensions import override
from zerver.lib.markdown import ResultWithFamily, walk_tree_with_family
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITES

class NestedCodeBlocksRenderer(Extension):

    @override
    def extendMarkdown(self, md: markdown.Markdown) -> None:
        if False:
            i = 10
            return i + 15
        md.treeprocessors.register(NestedCodeBlocksRendererTreeProcessor(md, self.getConfigs()), 'nested_code_blocks', PREPROCESSOR_PRIORITES['nested_code_blocks'])

class NestedCodeBlocksRendererTreeProcessor(markdown.treeprocessors.Treeprocessor):

    def __init__(self, md: markdown.Markdown, config: Mapping[str, Any]) -> None:
        if False:
            print('Hello World!')
        super().__init__(md)

    @override
    def run(self, root: Element) -> None:
        if False:
            i = 10
            return i + 15
        code_tags = walk_tree_with_family(root, self.get_code_tags)
        nested_code_blocks = self.get_nested_code_blocks(code_tags)
        for block in nested_code_blocks:
            (tag, text) = block.result
            codehilite_block = self.get_codehilite_block(text)
            self.replace_element(block.family.grandparent, codehilite_block, block.family.parent)

    def get_code_tags(self, e: Element) -> Optional[Tuple[str, Optional[str]]]:
        if False:
            print('Hello World!')
        if e.tag == 'code':
            return (e.tag, e.text)
        return None

    def get_nested_code_blocks(self, code_tags: List[ResultWithFamily[Tuple[str, Optional[str]]]]) -> List[ResultWithFamily[Tuple[str, Optional[str]]]]:
        if False:
            return 10
        nested_code_blocks = []
        for code_tag in code_tags:
            parent: Any = code_tag.family.parent
            grandparent: Any = code_tag.family.grandparent
            if parent.tag == 'p' and grandparent.tag == 'li' and (parent.text is None) and (len(parent) == 1) and (sum((1 for text in parent.itertext())) == 1):
                nested_code_blocks.append(code_tag)
        return nested_code_blocks

    def get_codehilite_block(self, code_block_text: Optional[str]) -> Element:
        if False:
            while True:
                i = 10
        div = Element('div')
        div.set('class', 'codehilite')
        pre = SubElement(div, 'pre')
        pre.text = code_block_text
        return div

    def replace_element(self, parent: Optional[Element], replacement: Element, element_to_replace: Element) -> None:
        if False:
            i = 10
            return i + 15
        if parent is None:
            return
        for (index, child) in enumerate(parent):
            if child is element_to_replace:
                parent.insert(index, replacement)
                parent.remove(element_to_replace)

def makeExtension(*args: Any, **kwargs: str) -> NestedCodeBlocksRenderer:
    if False:
        return 10
    return NestedCodeBlocksRenderer(**kwargs)