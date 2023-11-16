"""
A generic HTML whitelisting engine, designed to accommodate subclassing to override
specific rules.
"""
import re
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from django.utils.html import escape
ALLOWED_URL_SCHEMES = ['http', 'https', 'ftp', 'mailto', 'tel']
PROTOCOL_RE = re.compile('^[a-z0-9][-+.a-z0-9]*:')

def check_url(url_string):
    if False:
        print('Hello World!')
    unescaped = url_string.lower()
    unescaped = unescaped.replace('&lt;', '<')
    unescaped = unescaped.replace('&gt;', '>')
    unescaped = unescaped.replace('&amp;', '&')
    unescaped = re.sub('[`\\000-\\040\\177-\\240\\s]+', '', unescaped)
    unescaped = unescaped.replace('�', '')
    if PROTOCOL_RE.match(unescaped):
        protocol = unescaped.split(':', 1)[0]
        if protocol not in ALLOWED_URL_SCHEMES:
            return None
    return url_string

def attribute_rule(allowed_attrs):
    if False:
        print('Hello World!')
    "\n    Generator for functions that can be used as entries in Whitelister.element_rules.\n    These functions accept a tag, and modify its attributes by looking each attribute\n    up in the 'allowed_attrs' dict defined here:\n    * if the lookup fails, drop the attribute\n    * if the lookup returns a callable, replace the attribute with the result of calling\n      it - for example `{'title': uppercase}` will replace 'title' with the result of\n      uppercasing the title. If the callable returns None, the attribute is dropped.\n    * if the lookup returns a truthy value, keep the attribute; if falsy, drop it\n    "

    def fn(tag):
        if False:
            return 10
        for (attr, val) in list(tag.attrs.items()):
            rule = allowed_attrs.get(attr)
            if rule:
                if callable(rule):
                    new_val = rule(val)
                    if new_val is None:
                        del tag[attr]
                    else:
                        tag[attr] = new_val
                else:
                    pass
            else:
                del tag[attr]
    return fn
allow_without_attributes = attribute_rule({})
DEFAULT_ELEMENT_RULES = {'[document]': allow_without_attributes, 'a': attribute_rule({'href': check_url}), 'b': allow_without_attributes, 'br': allow_without_attributes, 'div': allow_without_attributes, 'em': allow_without_attributes, 'h1': allow_without_attributes, 'h2': allow_without_attributes, 'h3': allow_without_attributes, 'h4': allow_without_attributes, 'h5': allow_without_attributes, 'h6': allow_without_attributes, 'hr': allow_without_attributes, 'i': allow_without_attributes, 'img': attribute_rule({'src': check_url, 'width': True, 'height': True, 'alt': True}), 'li': allow_without_attributes, 'ol': allow_without_attributes, 'p': allow_without_attributes, 'strong': allow_without_attributes, 'sub': allow_without_attributes, 'sup': allow_without_attributes, 'ul': allow_without_attributes}

class Whitelister:
    element_rules = DEFAULT_ELEMENT_RULES

    def clean(self, html):
        if False:
            print('Hello World!')
        'Clean up an HTML string to contain just the allowed elements /\n        attributes'
        doc = BeautifulSoup(html, 'html5lib')
        self.clean_node(doc, doc)
        return doc.decode(formatter=escape)

    def clean_node(self, doc, node):
        if False:
            while True:
                i = 10
        'Clean a BeautifulSoup document in-place'
        if isinstance(node, NavigableString):
            self.clean_string_node(doc, node)
        elif isinstance(node, Tag):
            self.clean_tag_node(doc, node)
        else:
            self.clean_unknown_node(doc, node)

    def clean_string_node(self, doc, node):
        if False:
            while True:
                i = 10
        if isinstance(node, Comment):
            node.extract()
            return
        pass

    def clean_tag_node(self, doc, tag):
        if False:
            print('Hello World!')
        for child in list(tag.contents):
            self.clean_node(doc, child)
        try:
            rule = self.element_rules[tag.name]
        except KeyError:
            tag.unwrap()
            return
        rule(tag)

    def clean_unknown_node(self, doc, node):
        if False:
            while True:
                i = 10
        node.decompose()