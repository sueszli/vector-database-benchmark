"""
Functions for adding yaml encoding to the jinja context
"""
import io
import sys
import yaml
from salt.utils.decorators.jinja import jinja_filter

@jinja_filter()
def yaml_dquote(text):
    if False:
        i = 10
        return i + 15
    '\n    Make text into a double-quoted YAML string with correct escaping\n    for special characters.  Includes the opening and closing double\n    quote characters.\n    '
    with io.StringIO() as ostream:
        yemitter = yaml.emitter.Emitter(ostream, width=sys.maxsize)
        yemitter.write_double_quoted(str(text))
        return ostream.getvalue()

@jinja_filter()
def yaml_squote(text):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make text into a single-quoted YAML string with correct escaping\n    for special characters.  Includes the opening and closing single\n    quote characters.\n    '
    with io.StringIO() as ostream:
        yemitter = yaml.emitter.Emitter(ostream, width=sys.maxsize)
        yemitter.write_single_quoted(str(text))
        return ostream.getvalue()

@jinja_filter()
def yaml_encode(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    A simple YAML encode that can take a single-element datatype and return\n    a string representation.\n    '
    yrepr = yaml.representer.SafeRepresenter()
    ynode = yrepr.represent_data(data)
    if not isinstance(ynode, yaml.ScalarNode):
        raise TypeError('yaml_encode() only works with YAML scalar data; failed for {}'.format(type(data)))
    tag = ynode.tag.rsplit(':', 1)[-1]
    ret = ynode.value
    if tag == 'str':
        ret = yaml_dquote(ynode.value)
    return ret