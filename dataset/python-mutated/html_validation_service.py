"""HTML validation service."""
from __future__ import annotations
import json
import logging
from core import feconf
from core import utils
from core.constants import constants
from core.domain import fs_services
from core.domain import rte_component_registry
from extensions.objects.models import objects
from extensions.rich_text_components import components
import bs4
import defusedxml.ElementTree
from typing import Callable, Dict, Iterator, List, Tuple, Union

def wrap_with_siblings(tag: bs4.element.Tag, p: bs4.element.Tag) -> None:
    if False:
        return 10
    'This function wraps a tag and its unwrapped sibling in p tag.\n\n    Args:\n        tag: bs4.element.Tag. The tag which is to be wrapped in p tag\n            along with its unwrapped siblings.\n        p: bs4.element.Tag. The new p tag in soup in which the tag and\n            its siblings are to be wrapped.\n    '
    independent_parents = ['h1', 'p', 'pre', 'ol', 'ul', 'blockquote']
    prev_sib = list(tag.previous_siblings)
    next_sib = list(tag.next_siblings)
    index_of_first_unwrapped_sibling = -1
    for (index, sib) in enumerate(prev_sib):
        if sib.name in independent_parents:
            index_of_first_unwrapped_sibling = len(prev_sib) - index
            break
    for (index, sib) in enumerate(reversed(prev_sib)):
        if index >= index_of_first_unwrapped_sibling:
            sib.wrap(p)
    tag.wrap(p)
    for sib in next_sib:
        if sib.name not in independent_parents:
            sib.wrap(p)
        else:
            break
INLINE_COMPONENT_TAG_NAMES: List[str] = rte_component_registry.Registry.get_inline_component_tag_names()
BLOCK_COMPONENT_TAG_NAMES: List[str] = rte_component_registry.Registry.get_block_component_tag_names()
CHAR_MAPPINGS: List[Tuple[str, str]] = [(u'\xa0', u'\xa0'), (u'¡', u'¡'), (u'¢', u'¢'), (u'£', u'£'), (u'¤', u'¤'), (u'¥', u'¥'), (u'¦', u'¦'), (u'§', u'§'), (u'¨', u'¨'), (u'©', u'©'), (u'ª', u'ª'), (u'«', u'«'), (u'¬', u'¬'), (u'\xad', u'\xad'), (u'®', u'®'), (u'¯', u'¯'), (u'À', u'À'), (u'Á', u'Á'), (u'Â', u'Â'), (u'Ã', u'Ã'), (u'Ä', u'Ä'), (u'Å', u'Å'), (u'Æ', u'Æ'), (u'Ç', u'Ç'), (u'È', u'È'), (u'É', u'É'), (u'Ê', u'Ê'), (u'Ë', u'Ë'), (u'Ì', u'Ì'), (u'Í', u'Í'), (u'Î', u'Î'), (u'Ï', u'Ï'), (u'à', u'à'), (u'á', u'á'), (u'â', u'â'), (u'ã', u'ã'), (u'ä', u'ä'), (u'å', u'å'), (u'æ', u'æ'), (u'ç', u'ç'), (u'è', u'è'), (u'é', u'é'), (u'ê', u'ê'), (u'ë', u'ë'), (u'ì', u'ì'), (u'í', u'í'), (u'î', u'î'), (u'ï', u'ï'), (u'ð', u'ð'), (u'ñ', u'ñ'), (u'ò', u'ò'), (u'ó', u'ó'), (u'ô', u'ô'), (u'õ', u'õ'), (u'Â', ''), (u'àƒ', u'Ã'), (u'Ã\xa0', u'à'), (u'Ã¡', u'á'), (u'Ã¢', u'â'), (u'Ã£', u'ã'), (u'Ã¤', u'ä'), (u'Ã¥', u'å'), (u'Ã¦', u'æ'), (u'Ã§', u'ç'), (u'Ã¨', u'è'), (u'Ã©', u'é'), (u'Ãª', u'ê'), (u'Ã«', u'ë'), (u'Ã¬', u'ì'), (u'Ã\xad', u'í'), (u'Ã®', u'î'), (u'Ã¯', u'ï'), (u'Ã°', u'ð'), (u'Ã±', u'ñ'), (u'Ã²', u'ò'), (u'Ã³', u'ó'), (u'Ã´', u'ô'), (u'Ãµ', u'õ'), (u'Ã¶', u'ö'), (u'Ã·', u'÷'), (u'Ã¸', u'ø'), (u'Ã¹', u'ù'), (u'Ãº', u'ú'), (u'Ã»', u'û'), (u'Ã¼', u'ü'), (u'Ã½', u'ý'), (u'Ã¾', u'þ'), (u'Ã¿', u'ÿ'), (u'Ã–', u'Ö'), (u'Ã—', u'×'), (u'Ã‘', u'Ñ'), (u'Ã“', u'Ó'), (u'Ã„', u'Ä'), (u'Ã‡', u'Ç'), (u'Ã•', u'Õ'), (u'Ã€', u'À'), (u'Ãœ', u'Ü'), (u'ÃŸ', u'ß'), (u'ƒ\xa0', u''), (u'ÃŠ', u'Ê'), (u'Ãš', u'Ú'), (u'Ãƒ¡', u'á'), (u'Ãƒ¢', u'â'), (u'Ãƒ¤', u'ä'), (u'Ãƒ§', u'ç'), (u'Ãƒ¨', u'è'), (u'Ãƒ©', u'é'), (u'Ãƒª', u'ê'), (u'Ãƒ\xad', u'í'), (u'Ãƒ³', u'ó'), (u'Ãƒµ', u'õ'), (u'Ãƒ¶', u'ö'), (u'Ãƒº', u'ú'), (u'Ãƒ»', u'û'), (u'Ãƒ¼', u'ü'), (u'ÃƒÅ“', u'Ü'), (u'Ãƒâ€¢', u'Õ'), (u'Ã‚', u''), (u'Ã…Å¸', u'ş'), (u'Ã‰â€º', u'ɛ'), (u'Ã‰', u'É'), (u'Ã', u'à'), (u'Ä€', u'Ā'), (u'Ä…', u'ą'), (u'Ä‡', u'ć'), (u'Ä™', u'ę'), (u'ÄŒ', u'Č'), (u'Äž', u'Ğ'), (u'ÄŸ', u'ğ'), (u'ÄÅ¸', u'ğ'), (u'Ä«', u'ī'), (u'Ä°', u'İ'), (u'Ä±', u'ı'), (u'Ä»', u'Ļ'), (u'Åº', u'ź'), (u'Å¾', u'ž'), (u'Åž', u'Ş'), (u'Å›', u'ś'), (u'ÅŸ', u'ş'), (u'Å‘', u'ő'), (u'É›', u'ɛ'), (u'Ì€', u'̀'), (u'Î”', u'Δ'), (u'Ï€', u'π'), (u'Ñˆ', u'ш'), (u'×‘', u'ב'), (u'ØŸ', u'؟'), (u'Øµ', u'ص'), (u'Ø\xad', u'ح'), (u'Ø¤', u'ؤ'), (u'ÙŠ', u'ي'), (u'Ù…', u'م'), (u'Ùˆ', u'و'), (u'Ù‰', u'ى'), (u'à¶‡', u'ඇ'), (u'à¶…', u'අ'), (u'á¹›', u'ṛ'), (u'á»“', u'ồ'), (u'á»…', u'ễ'), (u'áº¿', u'ế'), (u'á»Ÿ', u'ở'), (u'â†’', u'→'), (u'âË†â€°', u'∉'), (u'â€œ', u'“'), (u'âˆ‰', u'∉'), (u'â…˜', u'⅘'), (u'â€™', u'’'), (u'âˆš', u'√'), (u'âˆˆ', u'∈'), (u'â…•', u'⅕'), (u'â…™', u'⅙'), (u'â€˜', u'‘'), (u'â€”', u'—'), (u'â€‹', u'\u200b'), (u'â€¦', u'…'), (u'â—¯', u'◯'), (u'â€“', u'–'), (u'â…–', u'⅖'), (u'â…”', u'⅔'), (u'â‰¤', u'≤'), (u'â‚¬', u'€'), (u'âœ…', u'✅'), (u'âž¤', u'➤'), (u'â˜º', u'☺'), (u'â›±', u'⛱'), (u'â€', u'†'), (u'â€“', u'–'), (u'â€¦', u'…'), (u'â¬…', u'⬅'), (u'ã‚Œ', u'れ'), (u'ã‚ˆ', u'よ'), (u'ã‚†', u'ゆ'), (u'ã‚‰', u'ら'), (u'ã‚€', u'む'), (u'ã‚„', u'や'), (u'ã‚“', u'ん'), (u'ã‚‚', u'も'), (u'ã‚’', u'を'), (u'ã‚Š', u'り'), (u'ä¸œ', u'东'), (u'åŒ—', u'北'), (u'åŽ»', u'去'), (u'æ“¦', u'擦'), (u'æœ¨', u'木'), (u'æˆ‘', u'我'), (u'æ˜¯', u'是'), (u'è¥¿', u'西'), (u'é”™', u'错'), (u'ï¼š', u'：'), (u'ï¼Ÿ', u'？'), (u'†“', u'–'), (u'†¦', u'…'), (u'채', u'ä'), (u'체', u'ü'), (u'覺', u'ı'), (u'카', u'ī'), (u'รง', u'ç'), (u'ร\x97', u'×'), (u'รท', u'÷'), (u'รถ', u'ö'), (u'รณ', u'ó'), (u'รป', u'û'), (u'ðŸ˜•', u'😕'), (u'ðŸ˜Š', u'😊'), (u'ðŸ˜‰', u'😉'), (u'ðŸ™„', u'🙄'), (u'ðŸ™‚', u'🙂'), (u'ğŸ˜Š', u'😊'), (u'ğŸ’¡', u'💡'), (u'ğŸ˜‘', u'😑'), (u'ğŸ˜Š', u'😊'), (u'ðŸ”–', u'🔖'), (u'ğŸ˜‰', u'😉'), (u'ðŸ˜ƒ', u'😃'), (u'ðŸ¤–', u'🤖'), (u'ðŸ“·', u'📷'), (u'ðŸ˜‚', u'😂'), (u'ðŸ“€', u'📀'), (u'ðŸ’¿', u'💿'), (u'ðŸ’¯', u'💯'), (u'ðŸ’¡', u'💡'), (u'ðŸ‘‹', u'👋'), (u'ðŸ˜±', u'😱'), (u'ðŸ˜‘', u'😑'), (u'ðŸ˜Š', u'😊'), (u'ðŸŽ§', u'🎧'), (u'ðŸŽ™', u'🎙'), (u'ðŸŽ¼', u'🎼'), (u'ðŸ“»', u'📻'), (u'ðŸ¤³', u'🤳'), (u'ðŸ‘Œ', u'👌'), (u'ðŸš¦', u'🚦'), (u'ðŸ¤—', u'🤗'), (u'ðŸ˜„', u'😄'), (u'ðŸ‘‰', u'👉'), (u'ðŸ“¡', u'📡'), (u'ðŸ“£', u'📣'), (u'ðŸ“¢', u'📢'), (u'ðŸ”Š', u'🔊'), (u'ðŸ˜Ž', u'😎'), (u'ðŸ˜‹', u'😋'), (u'ðŸ˜´', u'😴'), (u'ðŸ‘‘', u'👑'), (u'ðŸ‘†', u'👆'), (u'ðŸ‘®', u'👮'), (u'ðŸ“”', u'📔'), (u'ðŸ“¼', u'📼'), (u'ðŸ‡©', u'🇩'), (u'ðŸ‡ª', u'🇪'), (u'ðŸ‡¬', u'🇬'), (u'ðŸ‡§', u'🇧'), (u'ðŸ‡º', u'🇺'), (u'ðŸ‡¸', u'🇸'), (u'ðŸ•¶', u'🕶'), (u'ðŸ¤“', u'🤓'), (u'ðŸ¤”', u'🤔'), (u'ðŸ¤©', u'🤩'), (u'ðŸ¥º', u'🥺'), (u'ðŸ‘‰', u'\ud83d\udc49'), (u'ðŸ‘‰', u'\ud83d\udc49'), (u'\ud83d\udc49', u'👉'), (u'\t', u''), (u'\n', u''), (u'\xa0', u' ')]

def validate_rte_format(html_list: List[str], rte_format: str) -> Dict[str, List[str]]:
    if False:
        for i in range(10):
            print('nop')
    'This function checks if html strings in a given list are\n    valid for given RTE format.\n\n    Args:\n        html_list: list(str). List of html strings to be validated.\n        rte_format: str. The type of RTE for which html string is\n            to be validated.\n\n    Returns:\n        dict. Dictionary of all the error relations and strings.\n    '
    err_dict: Dict[str, List[str]] = {}
    err_dict['strings'] = []
    for html_data in html_list:
        soup_data = html_data
        soup = bs4.BeautifulSoup(soup_data.replace('<br>', '<br/>'), 'html.parser')
        is_invalid = validate_soup_for_rte(soup, rte_format, err_dict)
        if is_invalid:
            err_dict['strings'].append(html_data)
        for collapsible in soup.findAll(name='oppia-noninteractive-collapsible'):
            if 'content-with-value' not in collapsible.attrs or collapsible['content-with-value'] == '':
                is_invalid = True
            else:
                content_html = json.loads(utils.unescape_html(collapsible['content-with-value']))
                soup_for_collapsible = bs4.BeautifulSoup(content_html.replace('<br>', '<br/>'), 'html.parser')
                is_invalid = validate_soup_for_rte(soup_for_collapsible, rte_format, err_dict)
            if is_invalid:
                err_dict['strings'].append(html_data)
        for tabs in soup.findAll(name='oppia-noninteractive-tabs'):
            tab_content_json = utils.unescape_html(tabs['tab_contents-with-value'])
            tab_content_list = json.loads(tab_content_json)
            for tab_content in tab_content_list:
                content_html = tab_content['content']
                soup_for_tabs = bs4.BeautifulSoup(content_html.replace('<br>', '<br/>'), 'html.parser')
                is_invalid = validate_soup_for_rte(soup_for_tabs, rte_format, err_dict)
                if is_invalid:
                    err_dict['strings'].append(html_data)
    for key in err_dict:
        err_dict[key] = list(set(err_dict[key]))
    return err_dict

def validate_soup_for_rte(soup: bs4.BeautifulSoup, rte_format: str, err_dict: Dict[str, List[str]]) -> bool:
    if False:
        return 10
    'Validate content in given soup for given RTE format.\n\n    Args:\n        soup: bs4.BeautifulSoup. The html soup whose content is to be validated.\n        rte_format: str. The type of RTE for which html string is\n            to be validated.\n        err_dict: dict. The dictionary which stores invalid tags and strings.\n\n    Returns:\n        bool. Boolean indicating whether a html string is valid for given RTE.\n    '
    if rte_format == feconf.RTE_FORMAT_TEXTANGULAR:
        rte_type = 'RTE_TYPE_TEXTANGULAR'
    else:
        rte_type = 'RTE_TYPE_CKEDITOR'
    allowed_parent_list = feconf.RTE_CONTENT_SPEC[rte_type]['ALLOWED_PARENT_LIST']
    allowed_tag_list = feconf.RTE_CONTENT_SPEC[rte_type]['ALLOWED_TAG_LIST']
    is_invalid = False
    for content in soup.contents:
        if not content.name:
            is_invalid = True
    for tag in soup.findAll():
        if tag.name not in allowed_tag_list:
            if 'invalidTags' in err_dict:
                err_dict['invalidTags'].append(tag.name)
            else:
                err_dict['invalidTags'] = [tag.name]
            is_invalid = True
        parent = tag.parent.name
        if tag.name in allowed_tag_list and parent not in allowed_parent_list[tag.name]:
            if tag.name in err_dict:
                err_dict[tag.name].append(parent)
            else:
                err_dict[tag.name] = [parent]
            is_invalid = True
    return is_invalid

def validate_customization_args(html_list: List[str]) -> Dict[str, List[str]]:
    if False:
        i = 10
        return i + 15
    'Validates customization arguments of Rich Text Components in a list of\n    html string.\n\n    Args:\n        html_list: list(str). List of html strings to be validated.\n\n    Returns:\n        dict. Dictionary of all the invalid customisation args where\n        key is a Rich Text Component and value is the invalid html string.\n    '
    err_dict = {}
    rich_text_component_tag_names = INLINE_COMPONENT_TAG_NAMES + BLOCK_COMPONENT_TAG_NAMES
    tags_to_original_html_strings = {}
    for html_string in html_list:
        soup = bs4.BeautifulSoup(html_string, 'html.parser')
        for tag_name in rich_text_component_tag_names:
            for tag in soup.findAll(name=tag_name):
                tags_to_original_html_strings[tag] = html_string
    for (tag, html_string) in tags_to_original_html_strings.items():
        err_msg_list = list(validate_customization_args_in_tag(tag))
        for err_msg in err_msg_list:
            if err_msg:
                if err_msg not in err_dict:
                    err_dict[err_msg] = [html_string]
                elif html_string not in err_dict[err_msg]:
                    err_dict[err_msg].append(html_string)
    return err_dict

def validate_customization_args_in_tag(tag: bs4.element.Tag) -> Iterator[str]:
    if False:
        print('Hello World!')
    'Validates customization arguments of Rich Text Components in a soup.\n\n    Args:\n        tag: bs4.element.Tag. The html tag to be validated.\n\n    Yields:\n        str. Error message if the attributes of tag are invalid.\n    '
    component_types_to_component_classes = rte_component_registry.Registry.get_component_types_to_component_classes()
    simple_component_tag_names = rte_component_registry.Registry.get_simple_component_tag_names()
    tag_name = tag.name
    value_dict = {}
    attrs = tag.attrs
    for attr in attrs:
        value_dict[attr] = json.loads(utils.unescape_html(attrs[attr]))
    try:
        component_types_to_component_classes[tag_name].validate(value_dict)
        if tag_name == 'oppia-noninteractive-collapsible':
            content_html = value_dict['content-with-value']
            soup_for_collapsible = bs4.BeautifulSoup(content_html, 'html.parser')
            for component_name in simple_component_tag_names:
                for component_tag in soup_for_collapsible.findAll(name=component_name):
                    for err_msg in validate_customization_args_in_tag(component_tag):
                        yield err_msg
        elif tag_name == 'oppia-noninteractive-tabs':
            tab_content_list = value_dict['tab_contents-with-value']
            for tab_content in tab_content_list:
                content_html = tab_content['content']
                soup_for_tabs = bs4.BeautifulSoup(content_html, 'html.parser')
                for component_name in simple_component_tag_names:
                    for component_tag in soup_for_tabs.findAll(name=component_name):
                        for err_msg in validate_customization_args_in_tag(component_tag):
                            yield err_msg
    except Exception as e:
        yield str(e)

def validate_svg_filenames_in_math_rich_text(entity_type: str, entity_id: str, html_string: str) -> List[str]:
    if False:
        print('Hello World!')
    'Validates the SVG filenames for each math rich-text components and\n    returns a list of all invalid math tags in the given HTML.\n\n    Args:\n        entity_type: str. The type of the entity.\n        entity_id: str. The ID of the entity.\n        html_string: str. The HTML string.\n\n    Returns:\n        list(str). A list of invalid math tags in the HTML string.\n    '
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    error_list = []
    for math_tag in soup.findAll(name='oppia-noninteractive-math'):
        math_content_dict = json.loads(utils.unescape_html(math_tag['math_content-with-value']))
        svg_filename = objects.UnicodeString.normalize(math_content_dict['svg_filename'])
        if svg_filename == '':
            error_list.append(str(math_tag))
        else:
            fs = fs_services.GcsFileSystem(entity_type, entity_id)
            filepath = 'image/%s' % svg_filename
            if not fs.isfile(filepath):
                error_list.append(str(math_tag))
    return error_list

def validate_math_content_attribute_in_html(html_string: str) -> List[Dict[str, str]]:
    if False:
        while True:
            i = 10
    'Validates the format of SVG filenames for each math rich-text components\n    and returns a list of all invalid math tags in the given HTML.\n\n    Args:\n        html_string: str. The HTML string.\n\n    Returns:\n        list(dict(str, str)). A list of dicts each having the invalid tags in\n        the HTML string and the corresponding exception raised.\n    '
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    error_list = []
    for math_tag in soup.findAll(name='oppia-noninteractive-math'):
        math_content_dict = json.loads(utils.unescape_html(math_tag['math_content-with-value']))
        try:
            components.Math.validate({'math_content-with-value': math_content_dict})
        except utils.ValidationError as e:
            error_list.append({'invalid_tag': str(math_tag), 'error': str(e)})
    return error_list

def does_svg_tag_contains_xmlns_attribute(svg_string: Union[str, bytes]) -> bool:
    if False:
        i = 10
        return i + 15
    'Checks whether the svg tag in the given svg string contains the xmlns\n    attribute.\n\n    Args:\n        svg_string: str|bytes. The SVG string.\n\n    Returns:\n        bool. Whether the svg tag in the given svg string contains the xmlns\n        attribute.\n    '
    soup = bs4.BeautifulSoup(svg_string, 'html.parser')
    return all((svg_tag.get('xmlns') is not None for svg_tag in soup.findAll(name='svg')))

def get_invalid_svg_tags_and_attrs(svg_string: Union[str, bytes]) -> Tuple[List[str], List[str]]:
    if False:
        for i in range(10):
            print('nop')
    "Returns a set of all invalid tags and attributes for the provided SVG.\n\n    Args:\n        svg_string: str|bytes. The SVG string.\n\n    Returns:\n        tuple(list(str), list(str)). A 2-tuple, the first element of which\n        is a list of invalid tags, and the second element of which is a\n        list of invalid tag-specific attributes.\n        The format for the second element is <tag>:<attribute>, where the\n        <tag> represents the SVG tag for which the attribute is invalid\n        and <attribute> represents the invalid attribute.\n        eg. (['invalid-tag1', 'invalid-tag2'], ['path:invalid-attr'])\n    "
    soup = bs4.BeautifulSoup(svg_string, 'html.parser')
    invalid_elements = []
    invalid_attrs = []
    for element in soup.find_all():
        if element.name.lower() in constants.SVG_ATTRS_ALLOWLIST:
            for attr in element.attrs:
                if attr.lower() not in constants.SVG_ATTRS_ALLOWLIST[element.name.lower()]:
                    invalid_attrs.append('%s:%s' % (element.name, attr))
        else:
            invalid_elements.append(element.name)
    return (invalid_elements, invalid_attrs)

def check_for_svgdiagram_component_in_html(html_string: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Checks for existence of SvgDiagram component tags inside an HTML string.\n\n    Args:\n        html_string: str. HTML string to check.\n\n    Returns:\n        bool. Whether the given HTML string contains SvgDiagram component tag.\n    '
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    svgdiagram_tags = soup.findAll(name='oppia-noninteractive-svgdiagram')
    return bool(svgdiagram_tags)

def extract_svg_filenames_in_math_rte_components(html_string: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Extracts the svg_filenames from all the math-rich text components in\n    an HTML string.\n\n    Args:\n        html_string: str. The HTML string.\n\n    Returns:\n        list(str). A list of svg_filenames present in the HTML.\n    '
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    filenames = []
    for math_tag in soup.findAll(name='oppia-noninteractive-math'):
        math_content_dict = json.loads(utils.unescape_html(math_tag['math_content-with-value']))
        svg_filename = math_content_dict['svg_filename']
        if svg_filename != '':
            normalized_svg_filename = objects.UnicodeString.normalize(svg_filename)
            filenames.append(normalized_svg_filename)
    return filenames

def add_math_content_to_math_rte_components(html_string: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Replaces the attribute raw_latex-with-value in all Math component tags\n    with a new attribute math_content-with-value. The new attribute has an\n    additional field for storing SVG filenames. The field for SVG filename will\n    be an empty string.\n\n    Args:\n        html_string: str. HTML string to modify.\n\n    Returns:\n        str. Updated HTML string with all Math component tags having the new\n        attribute.\n\n    Raises:\n        Exception. Invalid latex string found while parsing the given\n            HTML string.\n    '
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    for math_tag in soup.findAll(name='oppia-noninteractive-math'):
        if math_tag.has_attr('raw_latex-with-value'):
            if not math_tag['raw_latex-with-value']:
                math_tag.decompose()
                continue
            try:
                raw_latex = json.loads(utils.unescape_html(math_tag['raw_latex-with-value']))
                normalized_raw_latex = objects.UnicodeString.normalize(raw_latex)
            except Exception as e:
                logging.exception('Invalid raw_latex string found in the math tag : %s' % str(e))
                raise e
            if math_tag.has_attr('svg_filename-with-value'):
                svg_filename = json.loads(utils.unescape_html(math_tag['svg_filename-with-value']))
                normalized_svg_filename = objects.UnicodeString.normalize(svg_filename)
                math_content_dict = {'raw_latex': normalized_raw_latex, 'svg_filename': normalized_svg_filename}
                del math_tag['svg_filename-with-value']
            else:
                math_content_dict = {'raw_latex': normalized_raw_latex, 'svg_filename': ''}
            normalized_math_content_dict = objects.MathExpressionContent.normalize(math_content_dict)
            math_tag['math_content-with-value'] = utils.escape_html(json.dumps(normalized_math_content_dict, sort_keys=True))
            del math_tag['raw_latex-with-value']
        elif math_tag.has_attr('math_content-with-value'):
            pass
        else:
            math_tag.decompose()
    return str(soup).replace('<br/>', '<br>')

def validate_math_tags_in_html(html_string: str) -> List[str]:
    if False:
        print('Hello World!')
    'Returns a list of all invalid math tags in the given HTML.\n\n    Args:\n        html_string: str. The HTML string.\n\n    Returns:\n        list(str). A list of invalid math tags in the HTML string.\n    '
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    error_list = []
    for math_tag in soup.findAll(name='oppia-noninteractive-math'):
        if math_tag.has_attr('raw_latex-with-value'):
            try:
                raw_latex = json.loads(utils.unescape_html(math_tag['raw_latex-with-value']))
                objects.UnicodeString.normalize(raw_latex)
            except Exception:
                error_list.append(math_tag)
        else:
            error_list.append(math_tag)
    return error_list

def validate_math_tags_in_html_with_attribute_math_content(html_string: str) -> List[str]:
    if False:
        return 10
    'Returns a list of all invalid new schema math tags in the given HTML.\n    The old schema has the attribute raw_latex-with-value while the new schema\n    has the attribute math-content-with-value which includes a field for storing\n    reference to SVGs.\n\n    Args:\n        html_string: str. The HTML string.\n\n    Returns:\n        list(str). A list of invalid math tags in the HTML string.\n    '
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    error_list = []
    for math_tag in soup.findAll(name='oppia-noninteractive-math'):
        if math_tag.has_attr('math_content-with-value'):
            try:
                math_content_dict = json.loads(utils.unescape_html(math_tag['math_content-with-value']))
                raw_latex = math_content_dict['raw_latex']
                svg_filename = math_content_dict['svg_filename']
                objects.UnicodeString.normalize(svg_filename)
                objects.UnicodeString.normalize(raw_latex)
            except Exception:
                error_list.append(math_tag)
        else:
            error_list.append(math_tag)
    return error_list

def is_parsable_as_xml(xml_string: bytes) -> bool:
    if False:
        print('Hello World!')
    'Checks if input string is parsable as XML.\n\n    Args:\n        xml_string: bytes. The XML string in bytes.\n\n    Returns:\n        bool. Whether xml_string is parsable as XML or not.\n    '
    if not isinstance(xml_string, bytes):
        return False
    try:
        defusedxml.ElementTree.fromstring(xml_string)
        return True
    except defusedxml.ElementTree.ParseError:
        return False

def convert_svg_diagram_to_image_for_soup(soup_context: bs4.BeautifulSoup) -> str:
    if False:
        print('Hello World!')
    '"Renames oppia-noninteractive-svgdiagram tag to\n    oppia-noninteractive-image and changes corresponding attributes for a given\n    soup context.\n\n    Args:\n        soup_context: bs4.BeautifulSoup. The bs4 soup context.\n\n    Returns:\n        str. The updated html string.\n    '
    for svg_image in soup_context.findAll(name='oppia-noninteractive-svgdiagram'):
        svg_filepath = svg_image['svg_filename-with-value']
        del svg_image['svg_filename-with-value']
        svg_image['filepath-with-value'] = svg_filepath
        svg_image['caption-with-value'] = utils.escape_html('""')
        svg_image.name = 'oppia-noninteractive-image'
    return str(soup_context)

def convert_svg_diagram_tags_to_image_tags(html_string: str) -> str:
    if False:
        return 10
    'Renames all the oppia-noninteractive-svgdiagram on the server to\n    oppia-noninteractive-image and changes corresponding attributes.\n\n    Args:\n        html_string: str. The HTML string to check.\n\n    Returns:\n        str. The updated html string.\n    '
    return str(_process_string_with_components(html_string, convert_svg_diagram_to_image_for_soup))

def _replace_incorrectly_encoded_chars(soup_context: bs4.BeautifulSoup) -> str:
    if False:
        return 10
    'Replaces incorrectly encoded character with the correct one in a given\n    HTML string.\n\n    Args:\n        soup_context: bs4.BeautifulSoup. The bs4 soup context.\n\n    Returns:\n        str. The updated html string.\n    '
    html_string = str(soup_context)
    char_mapping_tuples = CHAR_MAPPINGS + [(u'&nbsp;', u' ')]
    for (bad_char, good_char) in char_mapping_tuples:
        html_string = html_string.replace(bad_char, good_char)
    return html_string

def fix_incorrectly_encoded_chars(html_string: str) -> str:
    if False:
        return 10
    'Replaces incorrectly encoded character with the correct one in a given\n    HTML string.\n\n    Args:\n        html_string: str. The HTML string to modify.\n\n    Returns:\n        str. The updated html string.\n    '
    return str(_process_string_with_components(html_string, _replace_incorrectly_encoded_chars))

def _process_string_with_components(html_string: str, conversion_fn: Callable[[bs4.BeautifulSoup], str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Executes the provided conversion function after parsing complex RTE\n    components.\n\n    Args:\n        html_string: str. The HTML string to modify.\n        conversion_fn: function. The conversion function to be applied on\n            the HTML.\n\n    Returns:\n        str. The updated html string.\n    '
    soup = bs4.BeautifulSoup(html_string.encode(encoding='utf-8'), 'html.parser')
    for collapsible in soup.findAll(name='oppia-noninteractive-collapsible'):
        if 'content-with-value' in collapsible.attrs:
            content_html = json.loads(utils.unescape_html(collapsible['content-with-value']))
            soup_for_collapsible = bs4.BeautifulSoup(content_html.replace('<br>', '<br/>'), 'html.parser')
            collapsible['content-with-value'] = utils.escape_html(json.dumps(conversion_fn(soup_for_collapsible).replace('<br/>', '<br>')))
    for tabs in soup.findAll(name='oppia-noninteractive-tabs'):
        tab_content_json = utils.unescape_html(tabs['tab_contents-with-value'])
        tab_content_list = json.loads(tab_content_json)
        for tab_content in tab_content_list:
            content_html = tab_content['content']
            soup_for_tabs = bs4.BeautifulSoup(content_html.replace('<br>', '<br/>'), 'html.parser')
            tab_content['content'] = conversion_fn(soup_for_tabs).replace('<br/>', '<br>')
        tabs['tab_contents-with-value'] = utils.escape_html(json.dumps(tab_content_list))
    return conversion_fn(soup)