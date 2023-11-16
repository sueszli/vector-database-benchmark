"""
Construct one CLI page based on the input provided and returns customer choice
"""
import click

def do_paginate_cli(pages, page_to_be_rendered, items_per_page, is_last_page, cli_display_message):
    if False:
        print('Hello World!')
    '\n    Responsible for displaying a generic CLI page with available user choices for pagination/seletion\n    :param pages:\n    :param page_to_be_rendered:\n    :param items_per_page:\n    :param is_last_page:\n    :param cli_display_message:\n    :return: User decision on displayed page\n    '
    options = pages.get(page_to_be_rendered)
    choice_num = page_to_be_rendered * items_per_page + 1
    choices = []
    for option in options:
        msg = str(choice_num) + ' - ' + option
        click.echo('\t' + msg)
        choices.append(choice_num)
        choice_num = choice_num + 1
    if len(pages) == 1 and is_last_page:
        message = str.format(cli_display_message['single_page'])
    elif not page_to_be_rendered:
        choices = choices + ['N', 'n']
        message = cli_display_message['first_page']
    elif is_last_page and page_to_be_rendered == len(pages) - 1:
        choices = choices + ['P', 'p']
        message = cli_display_message['last_page']
    else:
        choices = choices + ['N', 'n', 'P', 'p']
        message = cli_display_message['middle_page']
    final_choices = list(map(str, choices))
    choice = click.prompt(message, type=click.Choice(final_choices), show_choices=False)
    if choice in ('N', 'n'):
        return {'choice': None, 'page_to_render': page_to_be_rendered + 1}
    if choice in ('P', 'p'):
        return {'choice': None, 'page_to_render': page_to_be_rendered - 1}
    index = int(choice) % items_per_page
    if index:
        index = index - 1
    else:
        index = items_per_page - 1
    return {'choice': options[index], 'page_to_render': None}