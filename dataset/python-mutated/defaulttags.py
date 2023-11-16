"""Default tags used by the template system, available to all templates."""
import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime
from itertools import cycle as itertools_cycle
from itertools import groupby
from django.conf import settings
from django.utils import timezone
from django.utils.html import conditional_escape, escape, format_html
from django.utils.itercompat import is_iterable
from django.utils.lorem_ipsum import paragraphs, words
from django.utils.safestring import mark_safe
from .base import BLOCK_TAG_END, BLOCK_TAG_START, COMMENT_TAG_END, COMMENT_TAG_START, FILTER_SEPARATOR, SINGLE_BRACE_END, SINGLE_BRACE_START, VARIABLE_ATTRIBUTE_SEPARATOR, VARIABLE_TAG_END, VARIABLE_TAG_START, Node, NodeList, TemplateSyntaxError, VariableDoesNotExist, kwarg_re, render_value_in_context, token_kwargs
from .context import Context
from .defaultfilters import date
from .library import Library
from .smartif import IfParser, Literal
register = Library()

class AutoEscapeControlNode(Node):
    """Implement the actions of the autoescape tag."""

    def __init__(self, setting, nodelist):
        if False:
            return 10
        self.setting = setting
        self.nodelist = nodelist

    def render(self, context):
        if False:
            return 10
        old_setting = context.autoescape
        context.autoescape = self.setting
        output = self.nodelist.render(context)
        context.autoescape = old_setting
        if self.setting:
            return mark_safe(output)
        else:
            return output

class CommentNode(Node):
    child_nodelists = ()

    def render(self, context):
        if False:
            return 10
        return ''

class CsrfTokenNode(Node):
    child_nodelists = ()

    def render(self, context):
        if False:
            i = 10
            return i + 15
        csrf_token = context.get('csrf_token')
        if csrf_token:
            if csrf_token == 'NOTPROVIDED':
                return format_html('')
            else:
                return format_html('<input type="hidden" name="csrfmiddlewaretoken" value="{}">', csrf_token)
        else:
            if settings.DEBUG:
                warnings.warn('A {% csrf_token %} was used in a template, but the context did not provide the value.  This is usually caused by not using RequestContext.')
            return ''

class CycleNode(Node):

    def __init__(self, cyclevars, variable_name=None, silent=False):
        if False:
            print('Hello World!')
        self.cyclevars = cyclevars
        self.variable_name = variable_name
        self.silent = silent

    def render(self, context):
        if False:
            i = 10
            return i + 15
        if self not in context.render_context:
            context.render_context[self] = itertools_cycle(self.cyclevars)
        cycle_iter = context.render_context[self]
        value = next(cycle_iter).resolve(context)
        if self.variable_name:
            context.set_upward(self.variable_name, value)
        if self.silent:
            return ''
        return render_value_in_context(value, context)

    def reset(self, context):
        if False:
            print('Hello World!')
        '\n        Reset the cycle iteration back to the beginning.\n        '
        context.render_context[self] = itertools_cycle(self.cyclevars)

class DebugNode(Node):

    def render(self, context):
        if False:
            print('Hello World!')
        if not settings.DEBUG:
            return ''
        from pprint import pformat
        output = [escape(pformat(val)) for val in context]
        output.append('\n\n')
        output.append(escape(pformat(sys.modules)))
        return ''.join(output)

class FilterNode(Node):

    def __init__(self, filter_expr, nodelist):
        if False:
            i = 10
            return i + 15
        self.filter_expr = filter_expr
        self.nodelist = nodelist

    def render(self, context):
        if False:
            while True:
                i = 10
        output = self.nodelist.render(context)
        with context.push(var=output):
            return self.filter_expr.resolve(context)

class FirstOfNode(Node):

    def __init__(self, variables, asvar=None):
        if False:
            return 10
        self.vars = variables
        self.asvar = asvar

    def render(self, context):
        if False:
            return 10
        first = ''
        for var in self.vars:
            value = var.resolve(context, ignore_failures=True)
            if value:
                first = render_value_in_context(value, context)
                break
        if self.asvar:
            context[self.asvar] = first
            return ''
        return first

class ForNode(Node):
    child_nodelists = ('nodelist_loop', 'nodelist_empty')

    def __init__(self, loopvars, sequence, is_reversed, nodelist_loop, nodelist_empty=None):
        if False:
            for i in range(10):
                print('nop')
        self.loopvars = loopvars
        self.sequence = sequence
        self.is_reversed = is_reversed
        self.nodelist_loop = nodelist_loop
        if nodelist_empty is None:
            self.nodelist_empty = NodeList()
        else:
            self.nodelist_empty = nodelist_empty

    def __repr__(self):
        if False:
            print('Hello World!')
        reversed_text = ' reversed' if self.is_reversed else ''
        return '<%s: for %s in %s, tail_len: %d%s>' % (self.__class__.__name__, ', '.join(self.loopvars), self.sequence, len(self.nodelist_loop), reversed_text)

    def render(self, context):
        if False:
            i = 10
            return i + 15
        if 'forloop' in context:
            parentloop = context['forloop']
        else:
            parentloop = {}
        with context.push():
            values = self.sequence.resolve(context, ignore_failures=True)
            if values is None:
                values = []
            if not hasattr(values, '__len__'):
                values = list(values)
            len_values = len(values)
            if len_values < 1:
                return self.nodelist_empty.render(context)
            nodelist = []
            if self.is_reversed:
                values = reversed(values)
            num_loopvars = len(self.loopvars)
            unpack = num_loopvars > 1
            loop_dict = context['forloop'] = {'parentloop': parentloop}
            for (i, item) in enumerate(values):
                loop_dict['counter0'] = i
                loop_dict['counter'] = i + 1
                loop_dict['revcounter'] = len_values - i
                loop_dict['revcounter0'] = len_values - i - 1
                loop_dict['first'] = i == 0
                loop_dict['last'] = i == len_values - 1
                pop_context = False
                if unpack:
                    try:
                        len_item = len(item)
                    except TypeError:
                        len_item = 1
                    if num_loopvars != len_item:
                        raise ValueError('Need {} values to unpack in for loop; got {}. '.format(num_loopvars, len_item))
                    unpacked_vars = dict(zip(self.loopvars, item))
                    pop_context = True
                    context.update(unpacked_vars)
                else:
                    context[self.loopvars[0]] = item
                for node in self.nodelist_loop:
                    nodelist.append(node.render_annotated(context))
                if pop_context:
                    context.pop()
        return mark_safe(''.join(nodelist))

class IfChangedNode(Node):
    child_nodelists = ('nodelist_true', 'nodelist_false')

    def __init__(self, nodelist_true, nodelist_false, *varlist):
        if False:
            while True:
                i = 10
        self.nodelist_true = nodelist_true
        self.nodelist_false = nodelist_false
        self._varlist = varlist

    def render(self, context):
        if False:
            print('Hello World!')
        state_frame = self._get_context_stack_frame(context)
        state_frame.setdefault(self)
        nodelist_true_output = None
        if self._varlist:
            compare_to = [var.resolve(context, ignore_failures=True) for var in self._varlist]
        else:
            compare_to = nodelist_true_output = self.nodelist_true.render(context)
        if compare_to != state_frame[self]:
            state_frame[self] = compare_to
            return nodelist_true_output or self.nodelist_true.render(context)
        elif self.nodelist_false:
            return self.nodelist_false.render(context)
        return ''

    def _get_context_stack_frame(self, context):
        if False:
            print('Hello World!')
        if 'forloop' in context:
            return context['forloop']
        else:
            return context.render_context

class IfNode(Node):

    def __init__(self, conditions_nodelists):
        if False:
            while True:
                i = 10
        self.conditions_nodelists = conditions_nodelists

    def __repr__(self):
        if False:
            return 10
        return '<%s>' % self.__class__.__name__

    def __iter__(self):
        if False:
            return 10
        for (_, nodelist) in self.conditions_nodelists:
            yield from nodelist

    @property
    def nodelist(self):
        if False:
            print('Hello World!')
        return NodeList(self)

    def render(self, context):
        if False:
            while True:
                i = 10
        for (condition, nodelist) in self.conditions_nodelists:
            if condition is not None:
                try:
                    match = condition.eval(context)
                except VariableDoesNotExist:
                    match = None
            else:
                match = True
            if match:
                return nodelist.render(context)
        return ''

class LoremNode(Node):

    def __init__(self, count, method, common):
        if False:
            return 10
        self.count = count
        self.method = method
        self.common = common

    def render(self, context):
        if False:
            i = 10
            return i + 15
        try:
            count = int(self.count.resolve(context))
        except (ValueError, TypeError):
            count = 1
        if self.method == 'w':
            return words(count, common=self.common)
        else:
            paras = paragraphs(count, common=self.common)
        if self.method == 'p':
            paras = ['<p>%s</p>' % p for p in paras]
        return '\n\n'.join(paras)
GroupedResult = namedtuple('GroupedResult', ['grouper', 'list'])

class RegroupNode(Node):

    def __init__(self, target, expression, var_name):
        if False:
            for i in range(10):
                print('nop')
        self.target = target
        self.expression = expression
        self.var_name = var_name

    def resolve_expression(self, obj, context):
        if False:
            i = 10
            return i + 15
        context[self.var_name] = obj
        return self.expression.resolve(context, ignore_failures=True)

    def render(self, context):
        if False:
            i = 10
            return i + 15
        obj_list = self.target.resolve(context, ignore_failures=True)
        if obj_list is None:
            context[self.var_name] = []
            return ''
        context[self.var_name] = [GroupedResult(grouper=key, list=list(val)) for (key, val) in groupby(obj_list, lambda obj: self.resolve_expression(obj, context))]
        return ''

class LoadNode(Node):
    child_nodelists = ()

    def render(self, context):
        if False:
            for i in range(10):
                print('nop')
        return ''

class NowNode(Node):

    def __init__(self, format_string, asvar=None):
        if False:
            print('Hello World!')
        self.format_string = format_string
        self.asvar = asvar

    def render(self, context):
        if False:
            print('Hello World!')
        tzinfo = timezone.get_current_timezone() if settings.USE_TZ else None
        formatted = date(datetime.now(tz=tzinfo), self.format_string)
        if self.asvar:
            context[self.asvar] = formatted
            return ''
        else:
            return formatted

class ResetCycleNode(Node):

    def __init__(self, node):
        if False:
            print('Hello World!')
        self.node = node

    def render(self, context):
        if False:
            for i in range(10):
                print('nop')
        self.node.reset(context)
        return ''

class SpacelessNode(Node):

    def __init__(self, nodelist):
        if False:
            for i in range(10):
                print('nop')
        self.nodelist = nodelist

    def render(self, context):
        if False:
            print('Hello World!')
        from django.utils.html import strip_spaces_between_tags
        return strip_spaces_between_tags(self.nodelist.render(context).strip())

class TemplateTagNode(Node):
    mapping = {'openblock': BLOCK_TAG_START, 'closeblock': BLOCK_TAG_END, 'openvariable': VARIABLE_TAG_START, 'closevariable': VARIABLE_TAG_END, 'openbrace': SINGLE_BRACE_START, 'closebrace': SINGLE_BRACE_END, 'opencomment': COMMENT_TAG_START, 'closecomment': COMMENT_TAG_END}

    def __init__(self, tagtype):
        if False:
            while True:
                i = 10
        self.tagtype = tagtype

    def render(self, context):
        if False:
            for i in range(10):
                print('nop')
        return self.mapping.get(self.tagtype, '')

class URLNode(Node):
    child_nodelists = ()

    def __init__(self, view_name, args, kwargs, asvar):
        if False:
            print('Hello World!')
        self.view_name = view_name
        self.args = args
        self.kwargs = kwargs
        self.asvar = asvar

    def __repr__(self):
        if False:
            return 10
        return "<%s view_name='%s' args=%s kwargs=%s as=%s>" % (self.__class__.__qualname__, self.view_name, repr(self.args), repr(self.kwargs), repr(self.asvar))

    def render(self, context):
        if False:
            return 10
        from django.urls import NoReverseMatch, reverse
        args = [arg.resolve(context) for arg in self.args]
        kwargs = {k: v.resolve(context) for (k, v) in self.kwargs.items()}
        view_name = self.view_name.resolve(context)
        try:
            current_app = context.request.current_app
        except AttributeError:
            try:
                current_app = context.request.resolver_match.namespace
            except AttributeError:
                current_app = None
        url = ''
        try:
            url = reverse(view_name, args=args, kwargs=kwargs, current_app=current_app)
        except NoReverseMatch:
            if self.asvar is None:
                raise
        if self.asvar:
            context[self.asvar] = url
            return ''
        else:
            if context.autoescape:
                url = conditional_escape(url)
            return url

class VerbatimNode(Node):

    def __init__(self, content):
        if False:
            return 10
        self.content = content

    def render(self, context):
        if False:
            for i in range(10):
                print('nop')
        return self.content

class WidthRatioNode(Node):

    def __init__(self, val_expr, max_expr, max_width, asvar=None):
        if False:
            while True:
                i = 10
        self.val_expr = val_expr
        self.max_expr = max_expr
        self.max_width = max_width
        self.asvar = asvar

    def render(self, context):
        if False:
            return 10
        try:
            value = self.val_expr.resolve(context)
            max_value = self.max_expr.resolve(context)
            max_width = int(self.max_width.resolve(context))
        except VariableDoesNotExist:
            return ''
        except (ValueError, TypeError):
            raise TemplateSyntaxError('widthratio final argument must be a number')
        try:
            value = float(value)
            max_value = float(max_value)
            ratio = value / max_value * max_width
            result = str(round(ratio))
        except ZeroDivisionError:
            result = '0'
        except (ValueError, TypeError, OverflowError):
            result = ''
        if self.asvar:
            context[self.asvar] = result
            return ''
        else:
            return result

class WithNode(Node):

    def __init__(self, var, name, nodelist, extra_context=None):
        if False:
            i = 10
            return i + 15
        self.nodelist = nodelist
        self.extra_context = extra_context or {}
        if name:
            self.extra_context[name] = var

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s>' % self.__class__.__name__

    def render(self, context):
        if False:
            while True:
                i = 10
        values = {key: val.resolve(context) for (key, val) in self.extra_context.items()}
        with context.push(**values):
            return self.nodelist.render(context)

@register.tag
def autoescape(parser, token):
    if False:
        for i in range(10):
            print('nop')
    '\n    Force autoescape behavior for this block.\n    '
    args = token.contents.split()
    if len(args) != 2:
        raise TemplateSyntaxError("'autoescape' tag requires exactly one argument.")
    arg = args[1]
    if arg not in ('on', 'off'):
        raise TemplateSyntaxError("'autoescape' argument should be 'on' or 'off'")
    nodelist = parser.parse(('endautoescape',))
    parser.delete_first_token()
    return AutoEscapeControlNode(arg == 'on', nodelist)

@register.tag
def comment(parser, token):
    if False:
        return 10
    '\n    Ignore everything between ``{% comment %}`` and ``{% endcomment %}``.\n    '
    parser.skip_past('endcomment')
    return CommentNode()

@register.tag
def cycle(parser, token):
    if False:
        print('Hello World!')
    '\n    Cycle among the given strings each time this tag is encountered.\n\n    Within a loop, cycles among the given strings each time through\n    the loop::\n\n        {% for o in some_list %}\n            <tr class="{% cycle \'row1\' \'row2\' %}">\n                ...\n            </tr>\n        {% endfor %}\n\n    Outside of a loop, give the values a unique name the first time you call\n    it, then use that name each successive time through::\n\n            <tr class="{% cycle \'row1\' \'row2\' \'row3\' as rowcolors %}">...</tr>\n            <tr class="{% cycle rowcolors %}">...</tr>\n            <tr class="{% cycle rowcolors %}">...</tr>\n\n    You can use any number of values, separated by spaces. Commas can also\n    be used to separate values; if a comma is used, the cycle values are\n    interpreted as literal strings.\n\n    The optional flag "silent" can be used to prevent the cycle declaration\n    from returning any value::\n\n        {% for o in some_list %}\n            {% cycle \'row1\' \'row2\' as rowcolors silent %}\n            <tr class="{{ rowcolors }}">{% include "subtemplate.html " %}</tr>\n        {% endfor %}\n    '
    args = token.split_contents()
    if len(args) < 2:
        raise TemplateSyntaxError("'cycle' tag requires at least two arguments")
    if len(args) == 2:
        name = args[1]
        if not hasattr(parser, '_named_cycle_nodes'):
            raise TemplateSyntaxError("No named cycles in template. '%s' is not defined" % name)
        if name not in parser._named_cycle_nodes:
            raise TemplateSyntaxError("Named cycle '%s' does not exist" % name)
        return parser._named_cycle_nodes[name]
    as_form = False
    if len(args) > 4:
        if args[-3] == 'as':
            if args[-1] != 'silent':
                raise TemplateSyntaxError("Only 'silent' flag is allowed after cycle's name, not '%s'." % args[-1])
            as_form = True
            silent = True
            args = args[:-1]
        elif args[-2] == 'as':
            as_form = True
            silent = False
    if as_form:
        name = args[-1]
        values = [parser.compile_filter(arg) for arg in args[1:-2]]
        node = CycleNode(values, name, silent=silent)
        if not hasattr(parser, '_named_cycle_nodes'):
            parser._named_cycle_nodes = {}
        parser._named_cycle_nodes[name] = node
    else:
        values = [parser.compile_filter(arg) for arg in args[1:]]
        node = CycleNode(values)
    parser._last_cycle_node = node
    return node

@register.tag
def csrf_token(parser, token):
    if False:
        while True:
            i = 10
    return CsrfTokenNode()

@register.tag
def debug(parser, token):
    if False:
        print('Hello World!')
    '\n    Output a whole load of debugging information, including the current\n    context and imported modules.\n\n    Sample usage::\n\n        <pre>\n            {% debug %}\n        </pre>\n    '
    return DebugNode()

@register.tag('filter')
def do_filter(parser, token):
    if False:
        i = 10
        return i + 15
    '\n    Filter the contents of the block through variable filters.\n\n    Filters can also be piped through each other, and they can have\n    arguments -- just like in variable syntax.\n\n    Sample usage::\n\n        {% filter force_escape|lower %}\n            This text will be HTML-escaped, and will appear in lowercase.\n        {% endfilter %}\n\n    Note that the ``escape`` and ``safe`` filters are not acceptable arguments.\n    Instead, use the ``autoescape`` tag to manage autoescaping for blocks of\n    template code.\n    '
    (_, rest) = token.contents.split(None, 1)
    filter_expr = parser.compile_filter('var|%s' % rest)
    for (func, unused) in filter_expr.filters:
        filter_name = getattr(func, '_filter_name', None)
        if filter_name in ('escape', 'safe'):
            raise TemplateSyntaxError('"filter %s" is not permitted.  Use the "autoescape" tag instead.' % filter_name)
    nodelist = parser.parse(('endfilter',))
    parser.delete_first_token()
    return FilterNode(filter_expr, nodelist)

@register.tag
def firstof(parser, token):
    if False:
        while True:
            i = 10
    '\n    Output the first variable passed that is not False.\n\n    Output nothing if all the passed variables are False.\n\n    Sample usage::\n\n        {% firstof var1 var2 var3 as myvar %}\n\n    This is equivalent to::\n\n        {% if var1 %}\n            {{ var1 }}\n        {% elif var2 %}\n            {{ var2 }}\n        {% elif var3 %}\n            {{ var3 }}\n        {% endif %}\n\n    but much cleaner!\n\n    You can also use a literal string as a fallback value in case all\n    passed variables are False::\n\n        {% firstof var1 var2 var3 "fallback value" %}\n\n    If you want to disable auto-escaping of variables you can use::\n\n        {% autoescape off %}\n            {% firstof var1 var2 var3 "<strong>fallback value</strong>" %}\n        {% autoescape %}\n\n    Or if only some variables should be escaped, you can use::\n\n        {% firstof var1 var2|safe var3 "<strong>fallback value</strong>"|safe %}\n    '
    bits = token.split_contents()[1:]
    asvar = None
    if not bits:
        raise TemplateSyntaxError("'firstof' statement requires at least one argument")
    if len(bits) >= 2 and bits[-2] == 'as':
        asvar = bits[-1]
        bits = bits[:-2]
    return FirstOfNode([parser.compile_filter(bit) for bit in bits], asvar)

@register.tag('for')
def do_for(parser, token):
    if False:
        for i in range(10):
            print('nop')
    '\n    Loop over each item in an array.\n\n    For example, to display a list of athletes given ``athlete_list``::\n\n        <ul>\n        {% for athlete in athlete_list %}\n            <li>{{ athlete.name }}</li>\n        {% endfor %}\n        </ul>\n\n    You can loop over a list in reverse by using\n    ``{% for obj in list reversed %}``.\n\n    You can also unpack multiple values from a two-dimensional array::\n\n        {% for key,value in dict.items %}\n            {{ key }}: {{ value }}\n        {% endfor %}\n\n    The ``for`` tag can take an optional ``{% empty %}`` clause that will\n    be displayed if the given array is empty or could not be found::\n\n        <ul>\n          {% for athlete in athlete_list %}\n            <li>{{ athlete.name }}</li>\n          {% empty %}\n            <li>Sorry, no athletes in this list.</li>\n          {% endfor %}\n        <ul>\n\n    The above is equivalent to -- but shorter, cleaner, and possibly faster\n    than -- the following::\n\n        <ul>\n          {% if athlete_list %}\n            {% for athlete in athlete_list %}\n              <li>{{ athlete.name }}</li>\n            {% endfor %}\n          {% else %}\n            <li>Sorry, no athletes in this list.</li>\n          {% endif %}\n        </ul>\n\n    The for loop sets a number of variables available within the loop:\n\n        ==========================  ================================================\n        Variable                    Description\n        ==========================  ================================================\n        ``forloop.counter``         The current iteration of the loop (1-indexed)\n        ``forloop.counter0``        The current iteration of the loop (0-indexed)\n        ``forloop.revcounter``      The number of iterations from the end of the\n                                    loop (1-indexed)\n        ``forloop.revcounter0``     The number of iterations from the end of the\n                                    loop (0-indexed)\n        ``forloop.first``           True if this is the first time through the loop\n        ``forloop.last``            True if this is the last time through the loop\n        ``forloop.parentloop``      For nested loops, this is the loop "above" the\n                                    current one\n        ==========================  ================================================\n    '
    bits = token.split_contents()
    if len(bits) < 4:
        raise TemplateSyntaxError("'for' statements should have at least four words: %s" % token.contents)
    is_reversed = bits[-1] == 'reversed'
    in_index = -3 if is_reversed else -2
    if bits[in_index] != 'in':
        raise TemplateSyntaxError("'for' statements should use the format 'for x in y': %s" % token.contents)
    invalid_chars = frozenset((' ', '"', "'", FILTER_SEPARATOR))
    loopvars = re.split(' *, *', ' '.join(bits[1:in_index]))
    for var in loopvars:
        if not var or not invalid_chars.isdisjoint(var):
            raise TemplateSyntaxError("'for' tag received an invalid argument: %s" % token.contents)
    sequence = parser.compile_filter(bits[in_index + 1])
    nodelist_loop = parser.parse(('empty', 'endfor'))
    token = parser.next_token()
    if token.contents == 'empty':
        nodelist_empty = parser.parse(('endfor',))
        parser.delete_first_token()
    else:
        nodelist_empty = None
    return ForNode(loopvars, sequence, is_reversed, nodelist_loop, nodelist_empty)

class TemplateLiteral(Literal):

    def __init__(self, value, text):
        if False:
            i = 10
            return i + 15
        self.value = value
        self.text = text

    def display(self):
        if False:
            print('Hello World!')
        return self.text

    def eval(self, context):
        if False:
            print('Hello World!')
        return self.value.resolve(context, ignore_failures=True)

class TemplateIfParser(IfParser):
    error_class = TemplateSyntaxError

    def __init__(self, parser, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.template_parser = parser
        super().__init__(*args, **kwargs)

    def create_var(self, value):
        if False:
            while True:
                i = 10
        return TemplateLiteral(self.template_parser.compile_filter(value), value)

@register.tag('if')
def do_if(parser, token):
    if False:
        print('Hello World!')
    '\n    Evaluate a variable, and if that variable is "true" (i.e., exists, is not\n    empty, and is not a false boolean value), output the contents of the block:\n\n    ::\n\n        {% if athlete_list %}\n            Number of athletes: {{ athlete_list|count }}\n        {% elif athlete_in_locker_room_list %}\n            Athletes should be out of the locker room soon!\n        {% else %}\n            No athletes.\n        {% endif %}\n\n    In the above, if ``athlete_list`` is not empty, the number of athletes will\n    be displayed by the ``{{ athlete_list|count }}`` variable.\n\n    The ``if`` tag may take one or several `` {% elif %}`` clauses, as well as\n    an ``{% else %}`` clause that will be displayed if all previous conditions\n    fail. These clauses are optional.\n\n    ``if`` tags may use ``or``, ``and`` or ``not`` to test a number of\n    variables or to negate a given variable::\n\n        {% if not athlete_list %}\n            There are no athletes.\n        {% endif %}\n\n        {% if athlete_list or coach_list %}\n            There are some athletes or some coaches.\n        {% endif %}\n\n        {% if athlete_list and coach_list %}\n            Both athletes and coaches are available.\n        {% endif %}\n\n        {% if not athlete_list or coach_list %}\n            There are no athletes, or there are some coaches.\n        {% endif %}\n\n        {% if athlete_list and not coach_list %}\n            There are some athletes and absolutely no coaches.\n        {% endif %}\n\n    Comparison operators are also available, and the use of filters is also\n    allowed, for example::\n\n        {% if articles|length >= 5 %}...{% endif %}\n\n    Arguments and operators _must_ have a space between them, so\n    ``{% if 1>2 %}`` is not a valid if tag.\n\n    All supported operators are: ``or``, ``and``, ``in``, ``not in``\n    ``==``, ``!=``, ``>``, ``>=``, ``<`` and ``<=``.\n\n    Operator precedence follows Python.\n    '
    bits = token.split_contents()[1:]
    condition = TemplateIfParser(parser, bits).parse()
    nodelist = parser.parse(('elif', 'else', 'endif'))
    conditions_nodelists = [(condition, nodelist)]
    token = parser.next_token()
    while token.contents.startswith('elif'):
        bits = token.split_contents()[1:]
        condition = TemplateIfParser(parser, bits).parse()
        nodelist = parser.parse(('elif', 'else', 'endif'))
        conditions_nodelists.append((condition, nodelist))
        token = parser.next_token()
    if token.contents == 'else':
        nodelist = parser.parse(('endif',))
        conditions_nodelists.append((None, nodelist))
        token = parser.next_token()
    if token.contents != 'endif':
        raise TemplateSyntaxError('Malformed template tag at line {}: "{}"'.format(token.lineno, token.contents))
    return IfNode(conditions_nodelists)

@register.tag
def ifchanged(parser, token):
    if False:
        i = 10
        return i + 15
    '\n    Check if a value has changed from the last iteration of a loop.\n\n    The ``{% ifchanged %}`` block tag is used within a loop. It has two\n    possible uses.\n\n    1. Check its own rendered contents against its previous state and only\n       displays the content if it has changed. For example, this displays a\n       list of days, only displaying the month if it changes::\n\n            <h1>Archive for {{ year }}</h1>\n\n            {% for date in days %}\n                {% ifchanged %}<h3>{{ date|date:"F" }}</h3>{% endifchanged %}\n                <a href="{{ date|date:"M/d"|lower }}/">{{ date|date:"j" }}</a>\n            {% endfor %}\n\n    2. If given one or more variables, check whether any variable has changed.\n       For example, the following shows the date every time it changes, while\n       showing the hour if either the hour or the date has changed::\n\n            {% for date in days %}\n                {% ifchanged date.date %} {{ date.date }} {% endifchanged %}\n                {% ifchanged date.hour date.date %}\n                    {{ date.hour }}\n                {% endifchanged %}\n            {% endfor %}\n    '
    bits = token.split_contents()
    nodelist_true = parser.parse(('else', 'endifchanged'))
    token = parser.next_token()
    if token.contents == 'else':
        nodelist_false = parser.parse(('endifchanged',))
        parser.delete_first_token()
    else:
        nodelist_false = NodeList()
    values = [parser.compile_filter(bit) for bit in bits[1:]]
    return IfChangedNode(nodelist_true, nodelist_false, *values)

def find_library(parser, name):
    if False:
        while True:
            i = 10
    try:
        return parser.libraries[name]
    except KeyError:
        raise TemplateSyntaxError("'%s' is not a registered tag library. Must be one of:\n%s" % (name, '\n'.join(sorted(parser.libraries))))

def load_from_library(library, label, names):
    if False:
        print('Hello World!')
    '\n    Return a subset of tags and filters from a library.\n    '
    subset = Library()
    for name in names:
        found = False
        if name in library.tags:
            found = True
            subset.tags[name] = library.tags[name]
        if name in library.filters:
            found = True
            subset.filters[name] = library.filters[name]
        if found is False:
            raise TemplateSyntaxError("'%s' is not a valid tag or filter in tag library '%s'" % (name, label))
    return subset

@register.tag
def load(parser, token):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load a custom template tag library into the parser.\n\n    For example, to load the template tags in\n    ``django/templatetags/news/photos.py``::\n\n        {% load news.photos %}\n\n    Can also be used to load an individual tag/filter from\n    a library::\n\n        {% load byline from news %}\n    '
    bits = token.contents.split()
    if len(bits) >= 4 and bits[-2] == 'from':
        name = bits[-1]
        lib = find_library(parser, name)
        subset = load_from_library(lib, name, bits[1:-2])
        parser.add_library(subset)
    else:
        for name in bits[1:]:
            lib = find_library(parser, name)
            parser.add_library(lib)
    return LoadNode()

@register.tag
def lorem(parser, token):
    if False:
        while True:
            i = 10
    '\n    Create random Latin text useful for providing test data in templates.\n\n    Usage format::\n\n        {% lorem [count] [method] [random] %}\n\n    ``count`` is a number (or variable) containing the number of paragraphs or\n    words to generate (default is 1).\n\n    ``method`` is either ``w`` for words, ``p`` for HTML paragraphs, ``b`` for\n    plain-text paragraph blocks (default is ``b``).\n\n    ``random`` is the word ``random``, which if given, does not use the common\n    paragraph (starting "Lorem ipsum dolor sit amet, consectetuer...").\n\n    Examples:\n\n    * ``{% lorem %}`` outputs the common "lorem ipsum" paragraph\n    * ``{% lorem 3 p %}`` outputs the common "lorem ipsum" paragraph\n      and two random paragraphs each wrapped in HTML ``<p>`` tags\n    * ``{% lorem 2 w random %}`` outputs two random latin words\n    '
    bits = list(token.split_contents())
    tagname = bits[0]
    common = bits[-1] != 'random'
    if not common:
        bits.pop()
    if bits[-1] in ('w', 'p', 'b'):
        method = bits.pop()
    else:
        method = 'b'
    if len(bits) > 1:
        count = bits.pop()
    else:
        count = '1'
    count = parser.compile_filter(count)
    if len(bits) != 1:
        raise TemplateSyntaxError('Incorrect format for %r tag' % tagname)
    return LoremNode(count, method, common)

@register.tag
def now(parser, token):
    if False:
        return 10
    '\n    Display the date, formatted according to the given string.\n\n    Use the same format as PHP\'s ``date()`` function; see https://php.net/date\n    for all the possible values.\n\n    Sample usage::\n\n        It is {% now "jS F Y H:i" %}\n    '
    bits = token.split_contents()
    asvar = None
    if len(bits) == 4 and bits[-2] == 'as':
        asvar = bits[-1]
        bits = bits[:-2]
    if len(bits) != 2:
        raise TemplateSyntaxError("'now' statement takes one argument")
    format_string = bits[1][1:-1]
    return NowNode(format_string, asvar)

@register.simple_tag(takes_context=True)
def query_string(context, query_dict=None, **kwargs):
    if False:
        return 10
    '\n    Add, remove, and change parameters of a ``QueryDict`` and return the result\n    as a query string. If the ``query_dict`` argument is not provided, default\n    to ``request.GET``.\n\n    For example::\n\n        {% query_string foo=3 %}\n\n    To remove a key::\n\n        {% query_string foo=None %}\n\n    To use with pagination::\n\n        {% query_string page=page_obj.next_page_number %}\n\n    A custom ``QueryDict`` can also be used::\n\n        {% query_string my_query_dict foo=3 %}\n    '
    if query_dict is None:
        query_dict = context.request.GET
    query_dict = query_dict.copy()
    for (key, value) in kwargs.items():
        if value is None:
            if key in query_dict:
                del query_dict[key]
        elif is_iterable(value) and (not isinstance(value, str)):
            query_dict.setlist(key, value)
        else:
            query_dict[key] = value
    if not query_dict:
        return ''
    query_string = query_dict.urlencode()
    return f'?{query_string}'

@register.tag
def regroup(parser, token):
    if False:
        i = 10
        return i + 15
    '\n    Regroup a list of alike objects by a common attribute.\n\n    This complex tag is best illustrated by use of an example: say that\n    ``musicians`` is a list of ``Musician`` objects that have ``name`` and\n    ``instrument`` attributes, and you\'d like to display a list that\n    looks like:\n\n        * Guitar:\n            * Django Reinhardt\n            * Emily Remler\n        * Piano:\n            * Lovie Austin\n            * Bud Powell\n        * Trumpet:\n            * Duke Ellington\n\n    The following snippet of template code would accomplish this dubious task::\n\n        {% regroup musicians by instrument as grouped %}\n        <ul>\n        {% for group in grouped %}\n            <li>{{ group.grouper }}\n            <ul>\n                {% for musician in group.list %}\n                <li>{{ musician.name }}</li>\n                {% endfor %}\n            </ul>\n        {% endfor %}\n        </ul>\n\n    As you can see, ``{% regroup %}`` populates a variable with a list of\n    objects with ``grouper`` and ``list`` attributes. ``grouper`` contains the\n    item that was grouped by; ``list`` contains the list of objects that share\n    that ``grouper``. In this case, ``grouper`` would be ``Guitar``, ``Piano``\n    and ``Trumpet``, and ``list`` is the list of musicians who play this\n    instrument.\n\n    Note that ``{% regroup %}`` does not work when the list to be grouped is not\n    sorted by the key you are grouping by! This means that if your list of\n    musicians was not sorted by instrument, you\'d need to make sure it is sorted\n    before using it, i.e.::\n\n        {% regroup musicians|dictsort:"instrument" by instrument as grouped %}\n    '
    bits = token.split_contents()
    if len(bits) != 6:
        raise TemplateSyntaxError("'regroup' tag takes five arguments")
    target = parser.compile_filter(bits[1])
    if bits[2] != 'by':
        raise TemplateSyntaxError("second argument to 'regroup' tag must be 'by'")
    if bits[4] != 'as':
        raise TemplateSyntaxError("next-to-last argument to 'regroup' tag must be 'as'")
    var_name = bits[5]
    expression = parser.compile_filter(var_name + VARIABLE_ATTRIBUTE_SEPARATOR + bits[3])
    return RegroupNode(target, expression, var_name)

@register.tag
def resetcycle(parser, token):
    if False:
        while True:
            i = 10
    '\n    Reset a cycle tag.\n\n    If an argument is given, reset the last rendered cycle tag whose name\n    matches the argument, else reset the last rendered cycle tag (named or\n    unnamed).\n    '
    args = token.split_contents()
    if len(args) > 2:
        raise TemplateSyntaxError('%r tag accepts at most one argument.' % args[0])
    if len(args) == 2:
        name = args[1]
        try:
            return ResetCycleNode(parser._named_cycle_nodes[name])
        except (AttributeError, KeyError):
            raise TemplateSyntaxError("Named cycle '%s' does not exist." % name)
    try:
        return ResetCycleNode(parser._last_cycle_node)
    except AttributeError:
        raise TemplateSyntaxError('No cycles in template.')

@register.tag
def spaceless(parser, token):
    if False:
        while True:
            i = 10
    '\n    Remove whitespace between HTML tags, including tab and newline characters.\n\n    Example usage::\n\n        {% spaceless %}\n            <p>\n                <a href="foo/">Foo</a>\n            </p>\n        {% endspaceless %}\n\n    This example returns this HTML::\n\n        <p><a href="foo/">Foo</a></p>\n\n    Only space between *tags* is normalized -- not space between tags and text.\n    In this example, the space around ``Hello`` isn\'t stripped::\n\n        {% spaceless %}\n            <strong>\n                Hello\n            </strong>\n        {% endspaceless %}\n    '
    nodelist = parser.parse(('endspaceless',))
    parser.delete_first_token()
    return SpacelessNode(nodelist)

@register.tag
def templatetag(parser, token):
    if False:
        i = 10
        return i + 15
    '\n    Output one of the bits used to compose template tags.\n\n    Since the template system has no concept of "escaping", to display one of\n    the bits used in template tags, you must use the ``{% templatetag %}`` tag.\n\n    The argument tells which template bit to output:\n\n        ==================  =======\n        Argument            Outputs\n        ==================  =======\n        ``openblock``       ``{%``\n        ``closeblock``      ``%}``\n        ``openvariable``    ``{{``\n        ``closevariable``   ``}}``\n        ``openbrace``       ``{``\n        ``closebrace``      ``}``\n        ``opencomment``     ``{#``\n        ``closecomment``    ``#}``\n        ==================  =======\n    '
    bits = token.contents.split()
    if len(bits) != 2:
        raise TemplateSyntaxError("'templatetag' statement takes one argument")
    tag = bits[1]
    if tag not in TemplateTagNode.mapping:
        raise TemplateSyntaxError("Invalid templatetag argument: '%s'. Must be one of: %s" % (tag, list(TemplateTagNode.mapping)))
    return TemplateTagNode(tag)

@register.tag
def url(parser, token):
    if False:
        return 10
    '\n    Return an absolute URL matching the given view with its parameters.\n\n    This is a way to define links that aren\'t tied to a particular URL\n    configuration::\n\n        {% url "url_name" arg1 arg2 %}\n\n        or\n\n        {% url "url_name" name1=value1 name2=value2 %}\n\n    The first argument is a URL pattern name. Other arguments are\n    space-separated values that will be filled in place of positional and\n    keyword arguments in the URL. Don\'t mix positional and keyword arguments.\n    All arguments for the URL must be present.\n\n    For example, if you have a view ``app_name.views.client_details`` taking\n    the client\'s id and the corresponding line in a URLconf looks like this::\n\n        path(\'client/<int:id>/\', views.client_details, name=\'client-detail-view\')\n\n    and this app\'s URLconf is included into the project\'s URLconf under some\n    path::\n\n        path(\'clients/\', include(\'app_name.urls\'))\n\n    then in a template you can create a link for a certain client like this::\n\n        {% url "client-detail-view" client.id %}\n\n    The URL will look like ``/clients/client/123/``.\n\n    The first argument may also be the name of a template variable that will be\n    evaluated to obtain the view name or the URL name, e.g.::\n\n        {% with url_name="client-detail-view" %}\n        {% url url_name client.id %}\n        {% endwith %}\n    '
    bits = token.split_contents()
    if len(bits) < 2:
        raise TemplateSyntaxError("'%s' takes at least one argument, a URL pattern name." % bits[0])
    viewname = parser.compile_filter(bits[1])
    args = []
    kwargs = {}
    asvar = None
    bits = bits[2:]
    if len(bits) >= 2 and bits[-2] == 'as':
        asvar = bits[-1]
        bits = bits[:-2]
    for bit in bits:
        match = kwarg_re.match(bit)
        if not match:
            raise TemplateSyntaxError('Malformed arguments to url tag')
        (name, value) = match.groups()
        if name:
            kwargs[name] = parser.compile_filter(value)
        else:
            args.append(parser.compile_filter(value))
    return URLNode(viewname, args, kwargs, asvar)

@register.tag
def verbatim(parser, token):
    if False:
        i = 10
        return i + 15
    "\n    Stop the template engine from rendering the contents of this block tag.\n\n    Usage::\n\n        {% verbatim %}\n            {% don't process this %}\n        {% endverbatim %}\n\n    You can also designate a specific closing tag block (allowing the\n    unrendered use of ``{% endverbatim %}``)::\n\n        {% verbatim myblock %}\n            ...\n        {% endverbatim myblock %}\n    "
    nodelist = parser.parse(('endverbatim',))
    parser.delete_first_token()
    return VerbatimNode(nodelist.render(Context()))

@register.tag
def widthratio(parser, token):
    if False:
        for i in range(10):
            print('nop')
    '\n    For creating bar charts and such. Calculate the ratio of a given value to a\n    maximum value, and then apply that ratio to a constant.\n\n    For example::\n\n        <img src="bar.png" alt="Bar"\n             height="10" width="{% widthratio this_value max_value max_width %}">\n\n    If ``this_value`` is 175, ``max_value`` is 200, and ``max_width`` is 100,\n    the image in the above example will be 88 pixels wide\n    (because 175/200 = .875; .875 * 100 = 87.5 which is rounded up to 88).\n\n    In some cases you might want to capture the result of widthratio in a\n    variable. It can be useful for instance in a blocktranslate like this::\n\n        {% widthratio this_value max_value max_width as width %}\n        {% blocktranslate %}The width is: {{ width }}{% endblocktranslate %}\n    '
    bits = token.split_contents()
    if len(bits) == 4:
        (tag, this_value_expr, max_value_expr, max_width) = bits
        asvar = None
    elif len(bits) == 6:
        (tag, this_value_expr, max_value_expr, max_width, as_, asvar) = bits
        if as_ != 'as':
            raise TemplateSyntaxError("Invalid syntax in widthratio tag. Expecting 'as' keyword")
    else:
        raise TemplateSyntaxError('widthratio takes at least three arguments')
    return WidthRatioNode(parser.compile_filter(this_value_expr), parser.compile_filter(max_value_expr), parser.compile_filter(max_width), asvar=asvar)

@register.tag('with')
def do_with(parser, token):
    if False:
        return 10
    '\n    Add one or more values to the context (inside of this block) for caching\n    and easy access.\n\n    For example::\n\n        {% with total=person.some_sql_method %}\n            {{ total }} object{{ total|pluralize }}\n        {% endwith %}\n\n    Multiple values can be added to the context::\n\n        {% with foo=1 bar=2 %}\n            ...\n        {% endwith %}\n\n    The legacy format of ``{% with person.some_sql_method as total %}`` is\n    still accepted.\n    '
    bits = token.split_contents()
    remaining_bits = bits[1:]
    extra_context = token_kwargs(remaining_bits, parser, support_legacy=True)
    if not extra_context:
        raise TemplateSyntaxError('%r expected at least one variable assignment' % bits[0])
    if remaining_bits:
        raise TemplateSyntaxError('%r received an invalid token: %r' % (bits[0], remaining_bits[0]))
    nodelist = parser.parse(('endwith',))
    parser.delete_first_token()
    return WithNode(None, None, nodelist, extra_context=extra_context)