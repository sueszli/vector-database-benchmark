"""Plugin to rewrite fields based on a given query."""
import shlex
from collections import defaultdict
import confuse
from beets import ui
from beets.dbcore import AndQuery, query_from_strings
from beets.library import Album, Item
from beets.plugins import BeetsPlugin

def rewriter(field, rules):
    if False:
        while True:
            i = 10
    'Template field function factory.\n\n    Create a template field function that rewrites the given field\n    with the given rewriting rules.\n    ``rules`` must be a list of (query, replacement) pairs.\n    '

    def fieldfunc(item):
        if False:
            i = 10
            return i + 15
        value = item._values_fixed[field]
        for (query, replacement) in rules:
            if query.match(item):
                return replacement
        return value
    return fieldfunc

class AdvancedRewritePlugin(BeetsPlugin):
    """Plugin to rewrite fields based on a given query."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        'Parse configuration and register template fields for rewriting.'
        super().__init__()
        template = confuse.Sequence({'match': str, 'field': str, 'replacement': str})
        rules = defaultdict(list)
        for rule in self.config.get(template):
            query = query_from_strings(AndQuery, Item, prefixes={}, query_parts=shlex.split(rule['match']))
            fieldname = rule['field']
            replacement = rule['replacement']
            if fieldname not in Item._fields:
                raise ui.UserError('invalid field name (%s) in rewriter' % fieldname)
            self._log.debug('adding template field {0} → {1}', fieldname, replacement)
            rules[fieldname].append((query, replacement))
            if fieldname == 'artist':
                rules['albumartist'].append((query, replacement))
        for (fieldname, fieldrules) in rules.items():
            getter = rewriter(fieldname, fieldrules)
            self.template_fields[fieldname] = getter
            if fieldname in Album._fields:
                self.album_template_fields[fieldname] = getter