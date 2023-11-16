"""
Tree processors manipulate the tree created by block processors. They can even create an entirely
new `ElementTree` object. This is an excellent place for creating summaries, adding collected
references, or last minute adjustments.

"""
from __future__ import annotations
import re
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Any
from . import util
from . import inlinepatterns
if TYPE_CHECKING:
    from markdown import Markdown

def build_treeprocessors(md: Markdown, **kwargs: Any) -> util.Registry[Treeprocessor]:
    if False:
        for i in range(10):
            print('nop')
    ' Build the default  `treeprocessors` for Markdown. '
    treeprocessors = util.Registry()
    treeprocessors.register(InlineProcessor(md), 'inline', 20)
    treeprocessors.register(PrettifyTreeprocessor(md), 'prettify', 10)
    treeprocessors.register(UnescapeTreeprocessor(md), 'unescape', 0)
    return treeprocessors

def isString(s: object) -> bool:
    if False:
        print('Hello World!')
    ' Return `True` if object is a string but not an  [`AtomicString`][markdown.util.AtomicString]. '
    if not isinstance(s, util.AtomicString):
        return isinstance(s, str)
    return False

class Treeprocessor(util.Processor):
    """
    `Treeprocessor`s are run on the `ElementTree` object before serialization.

    Each `Treeprocessor` implements a `run` method that takes a pointer to an
    `Element` and modifies it as necessary.

    `Treeprocessors` must extend `markdown.Treeprocessor`.

    """

    def run(self, root: etree.Element) -> etree.Element | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Subclasses of `Treeprocessor` should implement a `run` method, which\n        takes a root `Element`. This method can return another `Element`\n        object, and the existing root `Element` will be replaced, or it can\n        modify the current tree and return `None`.\n        '
        pass

class InlineProcessor(Treeprocessor):
    """
    A `Treeprocessor` that traverses a tree, applying inline patterns.
    """

    def __init__(self, md: Markdown):
        if False:
            i = 10
            return i + 15
        self.__placeholder_prefix = util.INLINE_PLACEHOLDER_PREFIX
        self.__placeholder_suffix = util.ETX
        self.__placeholder_length = 4 + len(self.__placeholder_prefix) + len(self.__placeholder_suffix)
        self.__placeholder_re = util.INLINE_PLACEHOLDER_RE
        self.md = md
        self.inlinePatterns = md.inlinePatterns
        self.ancestors: list[str] = []

    def __makePlaceholder(self, type: str) -> tuple[str, str]:
        if False:
            i = 10
            return i + 15
        ' Generate a placeholder '
        id = '%04d' % len(self.stashed_nodes)
        hash = util.INLINE_PLACEHOLDER % id
        return (hash, id)

    def __findPlaceholder(self, data: str, index: int) -> tuple[str | None, int]:
        if False:
            while True:
                i = 10
        '\n        Extract id from data string, start from index.\n\n        Arguments:\n            data: String.\n            index: Index, from which we start search.\n\n        Returns:\n            Placeholder id and string index, after the found placeholder.\n\n        '
        m = self.__placeholder_re.search(data, index)
        if m:
            return (m.group(1), m.end())
        else:
            return (None, index + 1)

    def __stashNode(self, node: etree.Element | str, type: str) -> str:
        if False:
            while True:
                i = 10
        ' Add node to stash. '
        (placeholder, id) = self.__makePlaceholder(type)
        self.stashed_nodes[id] = node
        return placeholder

    def __handleInline(self, data: str, patternIndex: int=0) -> str:
        if False:
            return 10
        '\n        Process string with inline patterns and replace it with placeholders.\n\n        Arguments:\n            data: A line of Markdown text.\n            patternIndex: The index of the `inlinePattern` to start with.\n\n        Returns:\n            String with placeholders.\n\n        '
        if not isinstance(data, util.AtomicString):
            startIndex = 0
            count = len(self.inlinePatterns)
            while patternIndex < count:
                (data, matched, startIndex) = self.__applyPattern(self.inlinePatterns[patternIndex], data, patternIndex, startIndex)
                if not matched:
                    patternIndex += 1
        return data

    def __processElementText(self, node: etree.Element, subnode: etree.Element, isText: bool=True) -> None:
        if False:
            print('Hello World!')
        "\n        Process placeholders in `Element.text` or `Element.tail`\n        of Elements popped from `self.stashed_nodes`.\n\n        Arguments:\n            node: Parent node.\n            subnode: Processing node.\n            isText: Boolean variable, True - it's text, False - it's a tail.\n\n        "
        if isText:
            text = subnode.text
            subnode.text = None
        else:
            text = subnode.tail
            subnode.tail = None
        childResult = self.__processPlaceholders(text, subnode, isText)
        if not isText and node is not subnode:
            pos = list(node).index(subnode) + 1
        else:
            pos = 0
        childResult.reverse()
        for newChild in childResult:
            node.insert(pos, newChild[0])

    def __processPlaceholders(self, data: str | None, parent: etree.Element, isText: bool=True) -> list[tuple[etree.Element, list[str]]]:
        if False:
            return 10
        "\n        Process string with placeholders and generate `ElementTree` tree.\n\n        Arguments:\n            data: String with placeholders instead of `ElementTree` elements.\n            parent: Element, which contains processing inline data.\n            isText: Boolean variable, True - it's text, False - it's a tail.\n\n        Returns:\n            List with `ElementTree` elements with applied inline patterns.\n\n        "

        def linkText(text: str | None) -> None:
            if False:
                print('Hello World!')
            if text:
                if result:
                    if result[-1][0].tail:
                        result[-1][0].tail += text
                    else:
                        result[-1][0].tail = text
                elif not isText:
                    if parent.tail:
                        parent.tail += text
                    else:
                        parent.tail = text
                elif parent.text:
                    parent.text += text
                else:
                    parent.text = text
        result = []
        strartIndex = 0
        while data:
            index = data.find(self.__placeholder_prefix, strartIndex)
            if index != -1:
                (id, phEndIndex) = self.__findPlaceholder(data, index)
                if id in self.stashed_nodes:
                    node = self.stashed_nodes.get(id)
                    if index > 0:
                        text = data[strartIndex:index]
                        linkText(text)
                    if not isinstance(node, str):
                        for child in [node] + list(node):
                            if child.tail:
                                if child.tail.strip():
                                    self.__processElementText(node, child, False)
                            if child.text:
                                if child.text.strip():
                                    self.__processElementText(child, child)
                    else:
                        linkText(node)
                        strartIndex = phEndIndex
                        continue
                    strartIndex = phEndIndex
                    result.append((node, self.ancestors[:]))
                else:
                    end = index + len(self.__placeholder_prefix)
                    linkText(data[strartIndex:end])
                    strartIndex = end
            else:
                text = data[strartIndex:]
                if isinstance(data, util.AtomicString):
                    text = util.AtomicString(text)
                linkText(text)
                data = ''
        return result

    def __applyPattern(self, pattern: inlinepatterns.Pattern, data: str, patternIndex: int, startIndex: int=0) -> tuple[str, bool, int]:
        if False:
            i = 10
            return i + 15
        '\n        Check if the line fits the pattern, create the necessary\n        elements, add it to `stashed_nodes`.\n\n        Arguments:\n            data: The text to be processed.\n            pattern: The pattern to be checked.\n            patternIndex: Index of current pattern.\n            startIndex: String index, from which we start searching.\n\n        Returns:\n            String with placeholders instead of `ElementTree` elements.\n\n        '
        new_style = isinstance(pattern, inlinepatterns.InlineProcessor)
        for exclude in pattern.ANCESTOR_EXCLUDES:
            if exclude.lower() in self.ancestors:
                return (data, False, 0)
        if new_style:
            match = None
            for match in pattern.getCompiledRegExp().finditer(data, startIndex):
                (node, start, end) = pattern.handleMatch(match, data)
                if start is None or end is None:
                    startIndex += match.end(0)
                    match = None
                    continue
                break
        else:
            match = pattern.getCompiledRegExp().match(data[startIndex:])
            leftData = data[:startIndex]
        if not match:
            return (data, False, 0)
        if not new_style:
            node = pattern.handleMatch(match)
            start = match.start(0)
            end = match.end(0)
        if node is None:
            return (data, True, end)
        if not isinstance(node, str):
            if not isinstance(node.text, util.AtomicString):
                for child in [node] + list(node):
                    if not isString(node):
                        if child.text:
                            self.ancestors.append(child.tag.lower())
                            child.text = self.__handleInline(child.text, patternIndex + 1)
                            self.ancestors.pop()
                        if child.tail:
                            child.tail = self.__handleInline(child.tail, patternIndex)
        placeholder = self.__stashNode(node, pattern.type())
        if new_style:
            return ('{}{}{}'.format(data[:start], placeholder, data[end:]), True, 0)
        else:
            return ('{}{}{}{}'.format(leftData, match.group(1), placeholder, match.groups()[-1]), True, 0)

    def __build_ancestors(self, parent: etree.Element | None, parents: list[str]) -> None:
        if False:
            while True:
                i = 10
        'Build the ancestor list.'
        ancestors = []
        while parent is not None:
            if parent is not None:
                ancestors.append(parent.tag.lower())
            parent = self.parent_map.get(parent)
        ancestors.reverse()
        parents.extend(ancestors)

    def run(self, tree: etree.Element, ancestors: list[str] | None=None) -> etree.Element:
        if False:
            i = 10
            return i + 15
        'Apply inline patterns to a parsed Markdown tree.\n\n        Iterate over `Element`, find elements with inline tag, apply inline\n        patterns and append newly created Elements to tree.  To avoid further\n        processing of string with inline patterns, instead of normal string,\n        use subclass [`AtomicString`][markdown.util.AtomicString]:\n\n            node.text = markdown.util.AtomicString("This will not be processed.")\n\n        Arguments:\n            tree: `Element` object, representing Markdown tree.\n            ancestors: List of parent tag names that precede the tree node (if needed).\n\n        Returns:\n            An element tree object with applied inline patterns.\n\n        '
        self.stashed_nodes: dict[str, etree.Element | str] = {}
        tree_parents = [] if ancestors is None else ancestors[:]
        self.parent_map = {c: p for p in tree.iter() for c in p}
        stack = [(tree, tree_parents)]
        while stack:
            (currElement, parents) = stack.pop()
            self.ancestors = parents
            self.__build_ancestors(currElement, self.ancestors)
            insertQueue = []
            for child in currElement:
                if child.text and (not isinstance(child.text, util.AtomicString)):
                    self.ancestors.append(child.tag.lower())
                    text = child.text
                    child.text = None
                    lst = self.__processPlaceholders(self.__handleInline(text), child)
                    for item in lst:
                        self.parent_map[item[0]] = child
                    stack += lst
                    insertQueue.append((child, lst))
                    self.ancestors.pop()
                if child.tail:
                    tail = self.__handleInline(child.tail)
                    dumby = etree.Element('d')
                    child.tail = None
                    tailResult = self.__processPlaceholders(tail, dumby, False)
                    if dumby.tail:
                        child.tail = dumby.tail
                    pos = list(currElement).index(child) + 1
                    tailResult.reverse()
                    for newChild in tailResult:
                        self.parent_map[newChild[0]] = currElement
                        currElement.insert(pos, newChild[0])
                if len(child):
                    self.parent_map[child] = currElement
                    stack.append((child, self.ancestors[:]))
            for (element, lst) in insertQueue:
                for (i, obj) in enumerate(lst):
                    newChild = obj[0]
                    element.insert(i, newChild)
        return tree

class PrettifyTreeprocessor(Treeprocessor):
    """ Add line breaks to the html document. """

    def _prettifyETree(self, elem: etree.Element) -> None:
        if False:
            while True:
                i = 10
        ' Recursively add line breaks to `ElementTree` children. '
        i = '\n'
        if self.md.is_block_level(elem.tag) and elem.tag not in ['code', 'pre']:
            if (not elem.text or not elem.text.strip()) and len(elem) and self.md.is_block_level(elem[0].tag):
                elem.text = i
            for e in elem:
                if self.md.is_block_level(e.tag):
                    self._prettifyETree(e)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i

    def run(self, root: etree.Element) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Add line breaks to `Element` object and its children. '
        self._prettifyETree(root)
        brs = root.iter('br')
        for br in brs:
            if not br.tail or not br.tail.strip():
                br.tail = '\n'
            else:
                br.tail = '\n%s' % br.tail
        pres = root.iter('pre')
        for pre in pres:
            if len(pre) and pre[0].tag == 'code':
                code = pre[0]
                if not len(code) and code.text is not None:
                    code.text = util.AtomicString(code.text.rstrip() + '\n')

class UnescapeTreeprocessor(Treeprocessor):
    """ Restore escaped chars """
    RE = re.compile('{}(\\d+){}'.format(util.STX, util.ETX))

    def _unescape(self, m: re.Match[str]) -> str:
        if False:
            return 10
        return chr(int(m.group(1)))

    def unescape(self, text: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.RE.sub(self._unescape, text)

    def run(self, root: etree.Element) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Loop over all elements and unescape all text. '
        for elem in root.iter():
            if elem.text and (not elem.tag == 'code'):
                elem.text = self.unescape(elem.text)
            if elem.tail:
                elem.tail = self.unescape(elem.tail)
            for (key, value) in elem.items():
                elem.set(key, self.unescape(value))