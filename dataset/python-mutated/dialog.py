"""
Provides utilities for reading dialog files and rendering dialogs populated
with custom data.
"""
import random
import os
import re
from pathlib import Path
from os.path import join
from mycroft.util import resolve_resource_file
from mycroft.util.format import expand_options
from mycroft.util.log import LOG

class MustacheDialogRenderer:
    """A dialog template renderer based on the mustache templating language."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.templates = {}
        self.recent_phrases = []
        self.max_recent_phrases = 3
        self.loop_prevention_offset = 2

    def load_template_file(self, template_name, filename):
        if False:
            print('Hello World!')
        'Load a template by file name into the templates cache.\n\n        Args:\n            template_name (str): a unique identifier for a group of templates\n            filename (str): a fully qualified filename of a mustache template.\n        '
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                template_text = line.strip()
                if not template_text.startswith('#') and template_text != '':
                    if template_name not in self.templates:
                        self.templates[template_name] = []
                    template_text = re.sub('\\{\\{+\\s*(.*?)\\s*\\}\\}+', '{\\1}', template_text)
                    self.templates[template_name].append(template_text)

    def render(self, template_name, context=None, index=None):
        if False:
            i = 10
            return i + 15
        '\n        Given a template name, pick a template and render it using the context.\n        If no matching template exists use template_name as template.\n\n        Tries not to let Mycroft say exactly the same thing twice in a row.\n\n        Args:\n            template_name (str): the name of a template group.\n            context (dict): dictionary representing values to be rendered\n            index (int): optional, the specific index in the collection of\n                templates\n\n        Returns:\n            str: the rendered string\n        '
        context = context or {}
        if template_name not in self.templates:
            return template_name.replace('.', ' ')
        template_functions = self.templates.get(template_name)
        if index is None:
            template_functions = [t for t in template_functions if t not in self.recent_phrases] or template_functions
            line = random.choice(template_functions)
        else:
            line = template_functions[index % len(template_functions)]
        line = line.format(**context)
        line = random.choice(expand_options(line))
        self.recent_phrases.append(line)
        if len(self.recent_phrases) > min(self.max_recent_phrases, len(self.templates.get(template_name)) - self.loop_prevention_offset):
            self.recent_phrases.pop(0)
        return line

def load_dialogs(dialog_dir, renderer=None):
    if False:
        while True:
            i = 10
    'Load all dialog files within the specified directory.\n\n    Args:\n        dialog_dir (str): directory that contains dialog files\n\n    Returns:\n        a loaded instance of a dialog renderer\n    '
    if renderer is None:
        renderer = MustacheDialogRenderer()
    directory = Path(dialog_dir)
    if not directory.exists() or not directory.is_dir():
        LOG.warning('No dialog files found: {}'.format(dialog_dir))
        return renderer
    for (path, _, files) in os.walk(str(directory)):
        for f in files:
            if f.endswith('.dialog'):
                renderer.load_template_file(f.replace('.dialog', ''), join(path, f))
    return renderer

def get(phrase, lang=None, context=None):
    if False:
        for i in range(10):
            print('nop')
    'Looks up a resource file for the given phrase.\n\n    If no file is found, the requested phrase is returned as the string. This\n    will use the default language for translations.\n\n    Args:\n        phrase (str): resource phrase to retrieve/translate\n        lang (str): the language to use\n        context (dict): values to be inserted into the string\n\n    Returns:\n        str: a randomized and/or translated version of the phrase\n    '
    if not lang:
        from mycroft.configuration import Configuration
        lang = Configuration.get().get('lang')
    filename = join('text', lang.lower(), phrase + '.dialog')
    template = resolve_resource_file(filename)
    if not template:
        LOG.debug('Resource file not found: {}'.format(filename))
        return phrase
    stache = MustacheDialogRenderer()
    stache.load_template_file('template', template)
    if not context:
        context = {}
    return stache.render('template', context)