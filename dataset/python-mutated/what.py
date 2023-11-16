import sys
import click
from rich.console import Console
from pywhat import __version__, identifier, printer
from pywhat.filter import Distribution, Filter
from pywhat.helper import AvailableTags, InvalidTag, Keys, str_to_key

def print_tags(ctx, opts, value):
    if False:
        print('Hello World!')
    if value:
        tags = sorted(AvailableTags().get_tags())
        console = Console()
        console.print('[bold #D7AFFF]' + '\n'.join(tags) + '[/bold #D7AFFF]')
        sys.exit()

def print_version(ctx, opts, value):
    if False:
        i = 10
        return i + 15
    if value:
        console = Console()
        console.print(f'PyWhat version [bold #49C3CE]{__version__}[/bold #49C3CE]')
        sys.exit()

def create_filter(rarity, include, exclude):
    if False:
        print('Hello World!')
    filters_dict = {}
    if rarity is not None:
        rarities = rarity.split(':')
        if len(rarities) != 2:
            print("Invalid rarity range format ('min:max' expected)")
            sys.exit(1)
        try:
            if not rarities[0].isspace() and rarities[0]:
                filters_dict['MinRarity'] = float(rarities[0])
            if not rarities[1].isspace() and rarities[1]:
                filters_dict['MaxRarity'] = float(rarities[1])
        except ValueError:
            print('Invalid rarity argument (float expected)')
            sys.exit(1)
    if include is not None:
        filters_dict['Tags'] = list(map(str.strip, include.split(',')))
    if exclude is not None:
        filters_dict['ExcludeTags'] = list(map(str.strip, exclude.split(',')))
    try:
        filter = Filter(filters_dict)
    except InvalidTag:
        print("Passed tags are not valid.\nYou can check available tags by using: 'pywhat --tags'")
        sys.exit(1)
    return filter

def get_text(ctx, opts, value):
    if False:
        for i in range(10):
            print('nop')
    if not value and (not click.get_text_stream('stdin').isatty()):
        return click.get_text_stream('stdin').read().strip()
    return value

@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument('text_input', callback=get_text, required=False)
@click.option('-t', '--tags', is_flag=True, expose_value=False, callback=print_tags, help='Show available tags and exit.')
@click.option('-r', '--rarity', help='Filter by rarity. Rarity is how unlikely something is to be a false-positive. The higher the number, the more unlikely. This is in the range of 0:1. To filter only items past 0.5, use 0.5: with the colon on the end. Default 0.1:1', default='0.1:1')
@click.option('-i', '--include', help='Only show matches with these tags.')
@click.option('-e', '--exclude', help='Exclude matches with these tags.')
@click.option('-o', '--only-text', is_flag=True, help='Do not scan files or folders.')
@click.option('-k', '--key', help='Sort by the specified key.')
@click.option('--reverse', is_flag=True, help='Sort in reverse order.')
@click.option('-br', '--boundaryless-rarity', help='Same as --rarity but for boundaryless mode (toggles what regexes will not have boundaries).', default='0.1:1')
@click.option('-bi', '--boundaryless-include', help='Same as --include but for boundaryless mode.')
@click.option('-be', '--boundaryless-exclude', help='Same as --exclude but for boundaryless mode.')
@click.option('-db', '--disable-boundaryless', is_flag=True, help='Disable boundaryless mode.')
@click.option('--json', is_flag=True, help='Return results in json format.')
@click.option('-v', '--version', is_flag=True, callback=print_version, help='Display the version of pywhat.')
@click.option('-if', '--include-filenames', is_flag=True, help='Search filenames for possible matches.')
@click.option('--format', required=False, help='Format output according to specified rules.')
@click.option('-pt', '--print-tags', is_flag=True, help='Add flags to output')
def main(**kwargs):
    if False:
        while True:
            i = 10
    '\n    pyWhat - Identify what something is.\n\n    Made by Bee https://twitter.com/bee_sec_san\n\n    https://github.com/bee-san\n\n    Filtration:\n\n        --rarity min:max\n\n            Rarity is how unlikely something is to be a false-positive. The higher the number, the more unlikely.\n\n            Only print entries with rarity in range [min,max]. min and max can be omitted.\n\n            Note: PyWhat by default has a rarity of 0.1. To see all matches, with many potential false positives use `0:`.\n\n        --include list\n\n            Only include entries containing at least one tag in a list. List is a comma separated list.\n\n        --exclude list\n\n            Exclude specified tags. List is a comma separated list.\n\n    Sorting:\n\n        --key key_name\n\n            Sort by the given key.\n\n        --reverse\n\n            Sort in reverse order.\n\n        Available keys:\n\n            name - Sort by the name of regex pattern\n\n            rarity - Sort by rarity\n\n            matched - Sort by a matched string\n\n            none - No sorting is done (the default)\n\n    Exporting:\n\n        --json\n\n            Return results in json format.\n\n    Boundaryless mode:\n\n        CLI tool matches strings like \'abcdTHM{hello}plze\' by default because the boundaryless mode is enabled for regexes with a rarity of 0.1 and higher.\n\n        Since boundaryless mode may produce a lot of false-positive matches, it is possible to disable it, either fully or partially.\n\n        \'--disable-boundaryless\' flag can be used to fully disable this mode.\n\n        In addition, \'-br\', \'-bi\', and \'-be\' options can be used to tweak which regexes should be in boundaryless mode.\n\n        Refer to the Filtration section for more information.\n\n    Formatting the output:\n\n        --format format_str\n\n            format_str can be equal to:\n\n                pretty - Output data in the table\n\n                json - Output data in json format\n\n                CUSTOM_STRING - Print data in the way you want. For every match CUSTOM_STRING will be printed and \'%x\' (See below for possible x values) will be substituted with a match value.\n\n                For example:\n\n                    pywhat --format \'%m - %n\' \'google.com htb{flag}\'\n\n                    will print:\n\n                    htb{flag} - HackTheBox Flag Format\n                    google.com - Uniform Resource Locator (URL)\n\n                Possible \'%x\' values:\n\n                    %m - matched text\n\n                    %n - name of regex\n\n                    %d - description (will not output if absent)\n\n                    %e - exploit (will not output if absent)\n\n                    %r - rarity\n\n                    %l - link (will not output if absent)\n\n                    %t - tags (in \'tag1, tag2 ...\' format)\n\n                If you want to print \'%\' or \'\\\' character - escape it: \'\\%\', \'\\\\\'.\n\n    Examples:\n\n        * what \'HTB{this is a flag}\'\n\n        * what \'0x52908400098527886E0F7030069857D2E4169EE7\'\n\n        * what -- \'52.6169586, -1.9779857\'\n\n        * what --rarity 0.6: \'myEmail@host.org\'\n\n        * what --rarity 0: --include "credentials" --exclude "aws" \'James:SecretPassword\'\n\n        * what -br 0.6: -be URL \'123myEmail@host.org456\'\n\n    Your text must either be in quotation marks, or use the POSIX standard of "--" to mean "anything after -- is textual input".\n\n\n    pyWhat can also search files or even a whole directory with recursion:\n\n        * what \'secret.txt\'\n\n        * what \'this/is/a/path\'\n\n    '
    if kwargs['text_input'] is None:
        sys.exit("Text input expected. Run 'pywhat --help' for help")
    dist = Distribution(create_filter(kwargs['rarity'], kwargs['include'], kwargs['exclude']))
    if kwargs['disable_boundaryless']:
        boundaryless = Filter({'Tags': []})
    else:
        boundaryless = create_filter(kwargs['boundaryless_rarity'], kwargs['boundaryless_include'], kwargs['boundaryless_exclude'])
    what_obj = What_Object(dist)
    if kwargs['key'] is None:
        key = Keys.NONE
    else:
        try:
            key = str_to_key(kwargs['key'])
        except ValueError:
            print('Invalid key')
            sys.exit(1)
    identified_output = what_obj.what_is_this(kwargs['text_input'], kwargs['only_text'], key, kwargs['reverse'], boundaryless, kwargs['include_filenames'])
    p = printer.Printing()
    if kwargs['json'] or str(kwargs['format']).strip() == 'json':
        p.print_json(identified_output)
    elif str(kwargs['format']).strip() == 'pretty':
        p.pretty_print(identified_output, kwargs['text_input'], kwargs['print_tags'])
    elif kwargs['format'] is not None:
        p.format_print(identified_output, kwargs['format'])
    else:
        p.print_raw(identified_output, kwargs['text_input'], kwargs['print_tags'])

class What_Object:

    def __init__(self, distribution):
        if False:
            return 10
        self.id = identifier.Identifier(dist=distribution)

    def what_is_this(self, text: str, only_text: bool, key, reverse: bool, boundaryless: Filter, include_filenames: bool) -> dict:
        if False:
            print('Hello World!')
        '\n        Returns a Python dictionary of everything that has been identified\n        '
        return self.id.identify(text, only_text=only_text, key=key, reverse=reverse, boundaryless=boundaryless, include_filenames=include_filenames)
if __name__ == '__main__':
    main()