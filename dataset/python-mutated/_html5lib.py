__license__ = 'MIT'
__all__ = ['HTML5TreeBuilder']
import warnings
import re
from bs4.builder import DetectsXMLParsedAsHTML, PERMISSIVE, HTML, HTML_5, HTMLTreeBuilder
from bs4.element import NamespacedAttribute, nonwhitespace_re
import html5lib
from html5lib.constants import namespaces, prefixes
from bs4.element import Comment, Doctype, NavigableString, Tag
try:
    from html5lib.treebuilders import _base as treebuilder_base
    new_html5lib = False
except ImportError as e:
    from html5lib.treebuilders import base as treebuilder_base
    new_html5lib = True

class HTML5TreeBuilder(HTMLTreeBuilder):
    """Use html5lib to build a tree.

    Note that this TreeBuilder does not support some features common
    to HTML TreeBuilders. Some of these features could theoretically
    be implemented, but at the very least it's quite difficult,
    because html5lib moves the parse tree around as it's being built.

    * This TreeBuilder doesn't use different subclasses of NavigableString
      based on the name of the tag in which the string was found.

    * You can't use a SoupStrainer to parse only part of a document.
    """
    NAME = 'html5lib'
    features = [NAME, PERMISSIVE, HTML_5, HTML]
    TRACKS_LINE_NUMBERS = True

    def prepare_markup(self, markup, user_specified_encoding, document_declared_encoding=None, exclude_encodings=None):
        if False:
            while True:
                i = 10
        self.user_specified_encoding = user_specified_encoding
        if exclude_encodings:
            warnings.warn("You provided a value for exclude_encoding, but the html5lib tree builder doesn't support exclude_encoding.", stacklevel=3)
        DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(markup)
        yield (markup, None, None, False)

    def feed(self, markup):
        if False:
            print('Hello World!')
        if self.soup.parse_only is not None:
            warnings.warn("You provided a value for parse_only, but the html5lib tree builder doesn't support parse_only. The entire document will be parsed.", stacklevel=4)
        parser = html5lib.HTMLParser(tree=self.create_treebuilder)
        self.underlying_builder.parser = parser
        extra_kwargs = dict()
        if not isinstance(markup, str):
            if new_html5lib:
                extra_kwargs['override_encoding'] = self.user_specified_encoding
            else:
                extra_kwargs['encoding'] = self.user_specified_encoding
        doc = parser.parse(markup, **extra_kwargs)
        if isinstance(markup, str):
            doc.original_encoding = None
        else:
            original_encoding = parser.tokenizer.stream.charEncoding[0]
            if not isinstance(original_encoding, str):
                original_encoding = original_encoding.name
            doc.original_encoding = original_encoding
        self.underlying_builder.parser = None

    def create_treebuilder(self, namespaceHTMLElements):
        if False:
            i = 10
            return i + 15
        self.underlying_builder = TreeBuilderForHtml5lib(namespaceHTMLElements, self.soup, store_line_numbers=self.store_line_numbers)
        return self.underlying_builder

    def test_fragment_to_document(self, fragment):
        if False:
            while True:
                i = 10
        'See `TreeBuilder`.'
        return '<html><head></head><body>%s</body></html>' % fragment

class TreeBuilderForHtml5lib(treebuilder_base.TreeBuilder):

    def __init__(self, namespaceHTMLElements, soup=None, store_line_numbers=True, **kwargs):
        if False:
            print('Hello World!')
        if soup:
            self.soup = soup
        else:
            from bs4 import BeautifulSoup
            self.soup = BeautifulSoup('', 'html.parser', store_line_numbers=store_line_numbers, **kwargs)
        super(TreeBuilderForHtml5lib, self).__init__(namespaceHTMLElements)
        self.parser = None
        self.store_line_numbers = store_line_numbers

    def documentClass(self):
        if False:
            i = 10
            return i + 15
        self.soup.reset()
        return Element(self.soup, self.soup, None)

    def insertDoctype(self, token):
        if False:
            i = 10
            return i + 15
        name = token['name']
        publicId = token['publicId']
        systemId = token['systemId']
        doctype = Doctype.for_name_and_ids(name, publicId, systemId)
        self.soup.object_was_parsed(doctype)

    def elementClass(self, name, namespace):
        if False:
            for i in range(10):
                print('nop')
        kwargs = {}
        if self.parser and self.store_line_numbers:
            (sourceline, sourcepos) = self.parser.tokenizer.stream.position()
            kwargs['sourceline'] = sourceline
            kwargs['sourcepos'] = sourcepos - 1
        tag = self.soup.new_tag(name, namespace, **kwargs)
        return Element(tag, self.soup, namespace)

    def commentClass(self, data):
        if False:
            for i in range(10):
                print('nop')
        return TextNode(Comment(data), self.soup)

    def fragmentClass(self):
        if False:
            while True:
                i = 10
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup('', 'html.parser')
        self.soup.name = '[document_fragment]'
        return Element(self.soup, self.soup, None)

    def appendChild(self, node):
        if False:
            i = 10
            return i + 15
        self.soup.append(node.element)

    def getDocument(self):
        if False:
            while True:
                i = 10
        return self.soup

    def getFragment(self):
        if False:
            return 10
        return treebuilder_base.TreeBuilder.getFragment(self).element

    def testSerializer(self, element):
        if False:
            while True:
                i = 10
        from bs4 import BeautifulSoup
        rv = []
        doctype_re = re.compile('^(.*?)(?: PUBLIC "(.*?)"(?: "(.*?)")?| SYSTEM "(.*?)")?$')

        def serializeElement(element, indent=0):
            if False:
                while True:
                    i = 10
            if isinstance(element, BeautifulSoup):
                pass
            if isinstance(element, Doctype):
                m = doctype_re.match(element)
                if m:
                    name = m.group(1)
                    if m.lastindex > 1:
                        publicId = m.group(2) or ''
                        systemId = m.group(3) or m.group(4) or ''
                        rv.append('|%s<!DOCTYPE %s "%s" "%s">' % (' ' * indent, name, publicId, systemId))
                    else:
                        rv.append('|%s<!DOCTYPE %s>' % (' ' * indent, name))
                else:
                    rv.append('|%s<!DOCTYPE >' % (' ' * indent,))
            elif isinstance(element, Comment):
                rv.append('|%s<!-- %s -->' % (' ' * indent, element))
            elif isinstance(element, NavigableString):
                rv.append('|%s"%s"' % (' ' * indent, element))
            else:
                if element.namespace:
                    name = '%s %s' % (prefixes[element.namespace], element.name)
                else:
                    name = element.name
                rv.append('|%s<%s>' % (' ' * indent, name))
                if element.attrs:
                    attributes = []
                    for (name, value) in list(element.attrs.items()):
                        if isinstance(name, NamespacedAttribute):
                            name = '%s %s' % (prefixes[name.namespace], name.name)
                        if isinstance(value, list):
                            value = ' '.join(value)
                        attributes.append((name, value))
                    for (name, value) in sorted(attributes):
                        rv.append('|%s%s="%s"' % (' ' * (indent + 2), name, value))
                indent += 2
                for child in element.children:
                    serializeElement(child, indent)
        serializeElement(element, 0)
        return '\n'.join(rv)

class AttrList(object):

    def __init__(self, element):
        if False:
            for i in range(10):
                print('nop')
        self.element = element
        self.attrs = dict(self.element.attrs)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return list(self.attrs.items()).__iter__()

    def __setitem__(self, name, value):
        if False:
            return 10
        list_attr = self.element.cdata_list_attributes or {}
        if name in list_attr.get('*', []) or (self.element.name in list_attr and name in list_attr.get(self.element.name, [])):
            if not isinstance(value, list):
                value = nonwhitespace_re.findall(value)
        self.element[name] = value

    def items(self):
        if False:
            return 10
        return list(self.attrs.items())

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        return list(self.attrs.keys())

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.attrs)

    def __getitem__(self, name):
        if False:
            return 10
        return self.attrs[name]

    def __contains__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return name in list(self.attrs.keys())

class Element(treebuilder_base.Node):

    def __init__(self, element, soup, namespace):
        if False:
            return 10
        treebuilder_base.Node.__init__(self, element.name)
        self.element = element
        self.soup = soup
        self.namespace = namespace

    def appendChild(self, node):
        if False:
            return 10
        string_child = child = None
        if isinstance(node, str):
            string_child = child = node
        elif isinstance(node, Tag):
            child = node
        elif node.element.__class__ == NavigableString:
            string_child = child = node.element
            node.parent = self
        else:
            child = node.element
            node.parent = self
        if not isinstance(child, str) and child.parent is not None:
            node.element.extract()
        if string_child is not None and self.element.contents and (self.element.contents[-1].__class__ == NavigableString):
            old_element = self.element.contents[-1]
            new_element = self.soup.new_string(old_element + string_child)
            old_element.replace_with(new_element)
            self.soup._most_recent_element = new_element
        else:
            if isinstance(node, str):
                child = self.soup.new_string(node)
            if self.element.contents:
                most_recent_element = self.element._last_descendant(False)
            elif self.element.next_element is not None:
                most_recent_element = self.soup._last_descendant()
            else:
                most_recent_element = self.element
            self.soup.object_was_parsed(child, parent=self.element, most_recent_element=most_recent_element)

    def getAttributes(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.element, Comment):
            return {}
        return AttrList(self.element)

    def setAttributes(self, attributes):
        if False:
            print('Hello World!')
        if attributes is not None and len(attributes) > 0:
            converted_attributes = []
            for (name, value) in list(attributes.items()):
                if isinstance(name, tuple):
                    new_name = NamespacedAttribute(*name)
                    del attributes[name]
                    attributes[new_name] = value
            self.soup.builder._replace_cdata_list_attribute_values(self.name, attributes)
            for (name, value) in list(attributes.items()):
                self.element[name] = value
            self.soup.builder.set_up_substitutions(self.element)
    attributes = property(getAttributes, setAttributes)

    def insertText(self, data, insertBefore=None):
        if False:
            return 10
        text = TextNode(self.soup.new_string(data), self.soup)
        if insertBefore:
            self.insertBefore(text, insertBefore)
        else:
            self.appendChild(text)

    def insertBefore(self, node, refNode):
        if False:
            return 10
        index = self.element.index(refNode.element)
        if node.element.__class__ == NavigableString and self.element.contents and (self.element.contents[index - 1].__class__ == NavigableString):
            old_node = self.element.contents[index - 1]
            new_str = self.soup.new_string(old_node + node.element)
            old_node.replace_with(new_str)
        else:
            self.element.insert(index, node.element)
            node.parent = self

    def removeChild(self, node):
        if False:
            while True:
                i = 10
        node.element.extract()

    def reparentChildren(self, new_parent):
        if False:
            return 10
        "Move all of this tag's children into another tag."
        element = self.element
        new_parent_element = new_parent.element
        final_next_element = element.next_sibling
        new_parents_last_descendant = new_parent_element._last_descendant(False, False)
        if len(new_parent_element.contents) > 0:
            new_parents_last_child = new_parent_element.contents[-1]
            new_parents_last_descendant_next_element = new_parents_last_descendant.next_element
        else:
            new_parents_last_child = None
            new_parents_last_descendant_next_element = new_parent_element.next_element
        to_append = element.contents
        if len(to_append) > 0:
            first_child = to_append[0]
            if new_parents_last_descendant is not None:
                first_child.previous_element = new_parents_last_descendant
            else:
                first_child.previous_element = new_parent_element
            first_child.previous_sibling = new_parents_last_child
            if new_parents_last_descendant is not None:
                new_parents_last_descendant.next_element = first_child
            else:
                new_parent_element.next_element = first_child
            if new_parents_last_child is not None:
                new_parents_last_child.next_sibling = first_child
            last_childs_last_descendant = to_append[-1]._last_descendant(False, True)
            last_childs_last_descendant.next_element = new_parents_last_descendant_next_element
            if new_parents_last_descendant_next_element is not None:
                new_parents_last_descendant_next_element.previous_element = last_childs_last_descendant
            last_childs_last_descendant.next_sibling = None
        for child in to_append:
            child.parent = new_parent_element
            new_parent_element.contents.append(child)
        element.contents = []
        element.next_element = final_next_element

    def cloneNode(self):
        if False:
            i = 10
            return i + 15
        tag = self.soup.new_tag(self.element.name, self.namespace)
        node = Element(tag, self.soup, self.namespace)
        for (key, value) in self.attributes:
            node.attributes[key] = value
        return node

    def hasContent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.element.contents

    def getNameTuple(self):
        if False:
            while True:
                i = 10
        if self.namespace == None:
            return (namespaces['html'], self.name)
        else:
            return (self.namespace, self.name)
    nameTuple = property(getNameTuple)

class TextNode(Element):

    def __init__(self, element, soup):
        if False:
            for i in range(10):
                print('nop')
        treebuilder_base.Node.__init__(self, None)
        self.element = element
        self.soup = soup

    def cloneNode(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError