import os
import shutil
import string
from importlib import import_module
from pathlib import Path
from typing import Optional, cast
from urllib.parse import urlparse
import scrapy
from scrapy.commands import ScrapyCommand
from scrapy.exceptions import UsageError
from scrapy.utils.template import render_templatefile, string_camelcase

def sanitize_module_name(module_name):
    if False:
        for i in range(10):
            print('nop')
    "Sanitize the given module name, by replacing dashes and points\n    with underscores and prefixing it with a letter if it doesn't start\n    with one\n    "
    module_name = module_name.replace('-', '_').replace('.', '_')
    if module_name[0] not in string.ascii_letters:
        module_name = 'a' + module_name
    return module_name

def extract_domain(url):
    if False:
        for i in range(10):
            print('nop')
    'Extract domain name from URL string'
    o = urlparse(url)
    if o.scheme == '' and o.netloc == '':
        o = urlparse('//' + url.lstrip('/'))
    return o.netloc

def verify_url_scheme(url):
    if False:
        while True:
            i = 10
    'Check url for scheme and insert https if none found.'
    parsed = urlparse(url)
    if parsed.scheme == '' and parsed.netloc == '':
        parsed = urlparse('//' + url)._replace(scheme='https')
    return parsed.geturl()

class Command(ScrapyCommand):
    requires_project = False
    default_settings = {'LOG_ENABLED': False}

    def syntax(self):
        if False:
            i = 10
            return i + 15
        return '[options] <name> <domain>'

    def short_desc(self):
        if False:
            i = 10
            return i + 15
        return 'Generate new spider using pre-defined templates'

    def add_options(self, parser):
        if False:
            i = 10
            return i + 15
        ScrapyCommand.add_options(self, parser)
        parser.add_argument('-l', '--list', dest='list', action='store_true', help='List available templates')
        parser.add_argument('-e', '--edit', dest='edit', action='store_true', help='Edit spider after creating it')
        parser.add_argument('-d', '--dump', dest='dump', metavar='TEMPLATE', help='Dump template to standard output')
        parser.add_argument('-t', '--template', dest='template', default='basic', help='Uses a custom template.')
        parser.add_argument('--force', dest='force', action='store_true', help='If the spider already exists, overwrite it with the template')

    def run(self, args, opts):
        if False:
            return 10
        if opts.list:
            self._list_templates()
            return
        if opts.dump:
            template_file = self._find_template(opts.dump)
            if template_file:
                print(template_file.read_text(encoding='utf-8'))
            return
        if len(args) != 2:
            raise UsageError()
        (name, url) = args[0:2]
        url = verify_url_scheme(url)
        module = sanitize_module_name(name)
        if self.settings.get('BOT_NAME') == module:
            print('Cannot create a spider with the same name as your project')
            return
        if not opts.force and self._spider_exists(name):
            return
        template_file = self._find_template(opts.template)
        if template_file:
            self._genspider(module, name, url, opts.template, template_file)
            if opts.edit:
                self.exitcode = os.system(f'scrapy edit "{name}"')

    def _genspider(self, module, name, url, template_name, template_file):
        if False:
            print('Hello World!')
        'Generate the spider module, based on the given template'
        capitalized_module = ''.join((s.capitalize() for s in module.split('_')))
        domain = extract_domain(url)
        tvars = {'project_name': self.settings.get('BOT_NAME'), 'ProjectName': string_camelcase(self.settings.get('BOT_NAME')), 'module': module, 'name': name, 'url': url, 'domain': domain, 'classname': f'{capitalized_module}Spider'}
        if self.settings.get('NEWSPIDER_MODULE'):
            spiders_module = import_module(self.settings['NEWSPIDER_MODULE'])
            spiders_dir = Path(spiders_module.__file__).parent.resolve()
        else:
            spiders_module = None
            spiders_dir = Path('.')
        spider_file = f'{spiders_dir / module}.py'
        shutil.copyfile(template_file, spider_file)
        render_templatefile(spider_file, **tvars)
        print(f'Created spider {name!r} using template {template_name!r} ', end='' if spiders_module else '\n')
        if spiders_module:
            print(f'in module:\n  {spiders_module.__name__}.{module}')

    def _find_template(self, template: str) -> Optional[Path]:
        if False:
            while True:
                i = 10
        template_file = Path(self.templates_dir, f'{template}.tmpl')
        if template_file.exists():
            return template_file
        print(f'Unable to find template: {template}\n')
        print('Use "scrapy genspider --list" to see all available templates.')
        return None

    def _list_templates(self):
        if False:
            return 10
        print('Available templates:')
        for file in sorted(Path(self.templates_dir).iterdir()):
            if file.suffix == '.tmpl':
                print(f'  {file.stem}')

    def _spider_exists(self, name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not self.settings.get('NEWSPIDER_MODULE'):
            path = Path(name + '.py')
            if path.exists():
                print(f'{path.resolve()} already exists')
                return True
            return False
        assert self.crawler_process is not None, 'crawler_process must be set before calling run'
        try:
            spidercls = self.crawler_process.spider_loader.load(name)
        except KeyError:
            pass
        else:
            print(f'Spider {name!r} already exists in module:')
            print(f'  {spidercls.__module__}')
            return True
        spiders_module = import_module(self.settings['NEWSPIDER_MODULE'])
        spiders_dir = Path(cast(str, spiders_module.__file__)).parent
        spiders_dir_abs = spiders_dir.resolve()
        path = spiders_dir_abs / (name + '.py')
        if path.exists():
            print(f'{path} already exists')
            return True
        return False

    @property
    def templates_dir(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return str(Path(self.settings['TEMPLATES_DIR'] or Path(scrapy.__path__[0], 'templates'), 'spiders'))