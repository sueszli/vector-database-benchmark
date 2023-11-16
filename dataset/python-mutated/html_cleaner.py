"""HTML sanitizing service."""
from __future__ import annotations
import html
import json
import logging
import urllib
from core import utils
from core.constants import constants
from core.domain import rte_component_registry
import bleach
import bs4
from typing import Dict, Final, List, TypedDict, Union, cast

class ComponentsDict(TypedDict):
    """Dictionary that represents RTE Components."""
    id: str
    customization_args: Dict[str, Union[str, int, str, bool, Dict[str, str]]]

def filter_a(tag: str, name: str, value: str) -> bool:
    if False:
        return 10
    "Returns whether the described attribute of a tag should be\n    allowed.\n\n    Args:\n        tag: str. The name of the tag passed.\n        name: str. The name of the attribute.\n        value: str. The value of the attribute.\n\n    Returns:\n        bool. Whether the given attribute should be allowed.\n\n    Raises:\n        Exception. The 'tag' is not as expected.\n    "
    if tag != 'a':
        raise Exception('The filter_a method should only be used for a tags.')
    if name in ('title', 'target'):
        return True
    if name == 'href':
        url_components = urllib.parse.urlsplit(value)
        if url_components[0] in ['http', 'https']:
            return True
        logging.error('Found invalid URL href: %s' % value)
    return False
ATTRS_ALLOWLIST: Final = {'a': filter_a, 'b': [], 'blockquote': [], 'br': [], 'code': [], 'div': [], 'em': [], 'h1': [], 'hr': [], 'i': [], 'li': [], 'ol': [], 'p': [], 'pre': [], 'span': [], 'strong': [], 'table': ['border'], 'tbody': [], 'td': [], 'tr': [], 'u': [], 'ul': []}

def clean(user_submitted_html: str) -> str:
    if False:
        while True:
            i = 10
    'Cleans a piece of user submitted HTML.\n\n    This only allows HTML from a restricted set of tags, attrs and styles.\n\n    Args:\n        user_submitted_html: str. An untrusted HTML string.\n\n    Returns:\n        str. The HTML string that results after stripping out unrecognized tags\n        and attributes.\n    '
    oppia_custom_tags = rte_component_registry.Registry.get_tag_list_with_attrs()
    core_tags = ATTRS_ALLOWLIST.copy()
    core_tags.update(oppia_custom_tags)
    tag_names = list(core_tags.keys())
    return bleach.clean(user_submitted_html, tags=tag_names, attributes=core_tags, strip=True)

def strip_html_tags(html_string: str) -> str:
    if False:
        i = 10
        return i + 15
    'Strips all HTML markup from an HTML string.\n\n    Args:\n        html_string: str. An HTML string.\n\n    Returns:\n        str. The HTML string that results after all the tags and attributes are\n        stripped out.\n    '
    return bleach.clean(html_string, tags=[], attributes={}, strip=True)

def get_image_filenames_from_html_strings(html_strings: List[str]) -> List[str]:
    if False:
        print('Hello World!')
    'Extracts the image filename from the oppia-noninteractive-image and\n    oppia-noninteractive-math RTE component from all the html strings\n    passed in.\n\n    Args:\n        html_strings: list(str). List of HTML strings.\n\n    Returns:\n        list(str). List of image filenames from html_strings.\n    '
    all_rte_components = []
    filenames = []
    for html_string in html_strings:
        all_rte_components.extend(get_rte_components(html_string))
    for rte_comp in all_rte_components:
        if 'id' in rte_comp and rte_comp['id'] == 'oppia-noninteractive-image':
            filename = cast(str, rte_comp['customization_args']['filepath-with-value'])
            filenames.append(filename)
        elif 'id' in rte_comp and rte_comp['id'] == 'oppia-noninteractive-math':
            content_to_filename_dict = cast(Dict[str, str], rte_comp['customization_args']['math_content-with-value'])
            filename = content_to_filename_dict['svg_filename']
            filenames.append(filename)
    return list(set(filenames))

def get_rte_components(html_string: str) -> List[ComponentsDict]:
    if False:
        for i in range(10):
            print('nop')
    "Extracts the RTE components from an HTML string.\n\n    Args:\n        html_string: str. An HTML string.\n\n    Returns:\n        list(dict). A list of dictionaries, each representing an RTE component.\n        Each dict in the list contains:\n        - id: str. The name of the component, i.e. 'oppia-noninteractive-link'.\n        - customization_args: dict. Customization arg specs for the component.\n    "
    components: List[ComponentsDict] = []
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    oppia_custom_tag_attrs = rte_component_registry.Registry.get_tag_list_with_attrs()
    for (tag_name, tag_attrs) in oppia_custom_tag_attrs.items():
        component_tags = soup.find_all(name=tag_name)
        for component_tag in component_tags:
            customization_args = {}
            for attr in tag_attrs:
                attr_val = html.unescape(component_tag[attr])
                customization_args[attr] = json.loads(attr_val)
            component: ComponentsDict = {'id': tag_name, 'customization_args': customization_args}
            components.append(component)
    return components

def is_html_empty(html_str: str) -> bool:
    if False:
        return 10
    'Checks if the html is empty or not.\n\n    Args:\n        html_str: str. The html that needs to be validated.\n\n    Returns:\n        bool. Returns True if the html is empty.\n    '
    if html_str.strip() in ['&quot;&quot;', '\\"&quot;&quot;\\"']:
        return True
    html_val = utils.unescape_html(html_str)
    html_val = html_val.replace('<p>', '').replace('</p>', '').replace('<br>', '').replace('<i>', '').replace('</i>', '').replace('<span>', '').replace('</span>', '').replace('<b>', '').replace('</b>', '').replace('<ol>', '').replace('</ol>', '').replace('<ul>', '').replace('</ul>', '').replace('<h1>', '').replace('</h1>', '').replace('<h2>', '').replace('</h2>', '').replace('<h3>', '').replace('</h3>', '').replace('<h4>', '').replace('</h4>', '').replace('<h5>', '').replace('</h5>', '').replace('<h6>', '').replace('</h6>', '').replace('<li>', '').replace('</li>', '').replace('&nbsp;', '').replace('<em>', '').replace('</em>', '').replace('<strong>', '').replace('</strong>', '').replace('""', '').replace("''", '')
    if html_val.strip() == '':
        return True
    return False

def _raise_validation_errors_for_escaped_html_tag(tag: bs4.BeautifulSoup, attr: str, tag_name: str) -> None:
    if False:
        return 10
    'Raises validation for the errored escaped html tag.\n\n    Args:\n        tag: bs4.BeautifulSoup. The tag which needs to be validated.\n        attr: str. The attribute name that needs to be validated inside the tag.\n        tag_name: str. The tag name.\n\n    Raises:\n        ValidationError. Tag does not have the attribute.\n        ValidationError. Tag attribute is empty.\n    '
    if not tag.has_attr(attr):
        raise utils.ValidationError("%s tag does not have '%s' attribute." % (tag_name, attr))
    if is_html_empty(tag[attr]):
        raise utils.ValidationError("%s tag '%s' attribute should not be empty." % (tag_name, attr))

def _raise_validation_errors_for_unescaped_html_tag(tag: bs4.BeautifulSoup, attr: str, tag_name: str) -> None:
    if False:
        i = 10
        return i + 15
    'Raises validation for the errored unescaped html tag.\n\n    Args:\n        tag: bs4.BeautifulSoup. The tag which needs to be validated.\n        attr: str. The attribute name that needs to be validated inside the tag.\n        tag_name: str. The tag name.\n\n    Raises:\n        ValidationError. Tag does not have the attribute.\n        ValidationError. Tag attribute is empty.\n    '
    if not tag.has_attr(attr):
        raise utils.ValidationError("%s tag does not have '%s' attribute." % (tag_name, attr))
    attr_value = utils.unescape_html(tag[attr])[1:-1].replace('\\"', '')
    if is_html_empty(attr_value):
        raise utils.ValidationError("%s tag '%s' attribute should not be empty." % (tag_name, attr))

def validate_rte_tags(html_data: str, is_tag_nested_inside_tabs_or_collapsible: bool=False) -> None:
    if False:
        return 10
    'Validate all the RTE tags.\n\n    Args:\n        html_data: str. The RTE content of the state.\n        is_tag_nested_inside_tabs_or_collapsible: bool. True when we\n            validate tags inside `Tabs` or `Collapsible` tag.\n\n    Raises:\n        ValidationError. Image does not have alt-with-value attribute.\n        ValidationError. Image alt-with-value attribute have less\n            than 5 characters.\n        ValidationError. Image does not have caption-with-value attribute.\n        ValidationError. Image caption-with-value attribute have more\n            than 500 characters.\n        ValidationError. Image does not have filepath-with-value attribute.\n        ValidationError. Image filepath-with-value attribute should not be\n            empty.\n        ValidationError. SkillReview does not have text-with-value attribute.\n        ValidationError. SkillReview text-with-value attribute should not be\n            empty.\n        ValidationError. SkillReview does not have skill_id-with-value\n            attribute.\n        ValidationError. SkillReview skill_id-with-value attribute should not be\n            empty.\n        ValidationError. Video does not have start-with-value attribute.\n        ValidationError. Video start-with-value attribute should not be empty.\n        ValidationError. Video does not have end-with-value attribute.\n        ValidationError. Video end-with-value attribute should not be empty.\n        ValidationError. Start value is greater than end value.\n        ValidationError. Video does not have autoplay-with-value attribute.\n        ValidationError. Video autoplay-with-value attribute should be boolean.\n        ValidationError. Video does not have video_id-with-value attribute.\n        ValidationError. Link does not have text-with-value attribute.\n        ValidationError. Link does not have url-with-value attribute.\n        ValidationError. Link url-with-value attribute should not be empty.\n        ValidationError. Math does not have math_content-with-value attribute.\n        ValidationError. Math math_content-with-value attribute should not be\n            empty.\n        ValidationError. Math does not have raw_latex-with-value attribute.\n        ValidationError. Math raw_latex-with-value attribute should not be\n            empty.\n        ValidationError. Math does not have svg_filename-with-value attribute.\n        ValidationError. Math svg_filename-with-value attribute should not be\n            empty.\n        ValidationError. Math svg_filename attribute does not have svg\n            extension.\n        ValidationError. Tabs tag present inside another tabs or collapsible.\n        ValidationError. Collapsible tag present inside tabs or another\n            collapsible.\n    '
    soup = bs4.BeautifulSoup(html_data, 'html.parser')
    for tag in soup.find_all('oppia-noninteractive-image'):
        if not tag.has_attr('alt-with-value'):
            raise utils.ValidationError("Image tag does not have 'alt-with-value' attribute.")
        if not tag.has_attr('caption-with-value'):
            raise utils.ValidationError("Image tag does not have 'caption-with-value' attribute.")
        caption_value = utils.unescape_html(tag['caption-with-value'])[1:-1].replace('\\"', '')
        if len(caption_value.strip()) > 500:
            raise utils.ValidationError("Image tag 'caption-with-value' attribute should not be greater than 500 characters.")
        if not tag.has_attr('filepath-with-value'):
            raise utils.ValidationError("Image tag does not have 'filepath-with-value' attribute.")
        filepath_value = utils.unescape_html(tag['filepath-with-value'])[1:-1].replace('\\"', '')
        if is_html_empty(filepath_value):
            raise utils.ValidationError("Image tag 'filepath-with-value' attribute should not be empty.")
    for tag in soup.find_all('oppia-noninteractive-skillreview'):
        _raise_validation_errors_for_unescaped_html_tag(tag, 'text-with-value', 'SkillReview')
        _raise_validation_errors_for_unescaped_html_tag(tag, 'skill_id-with-value', 'SkillReview')
    for tag in soup.find_all('oppia-noninteractive-video'):
        _raise_validation_errors_for_escaped_html_tag(tag, 'start-with-value', 'Video')
        _raise_validation_errors_for_escaped_html_tag(tag, 'end-with-value', 'Video')
        start_value = float(tag['start-with-value'].strip())
        end_value = float(tag['end-with-value'].strip())
        if start_value > end_value and start_value != 0.0 and (end_value != 0.0):
            raise utils.ValidationError('Start value should not be greater than End value in Video tag.')
        if not tag.has_attr('autoplay-with-value'):
            raise utils.ValidationError("Video tag does not have 'autoplay-with-value' attribute.")
        if tag['autoplay-with-value'].strip() not in ('true', 'false', "'true'", "'false'", '"true"', '"false"', True, False):
            raise utils.ValidationError("Video tag 'autoplay-with-value' attribute should be a boolean value.")
        _raise_validation_errors_for_unescaped_html_tag(tag, 'video_id-with-value', 'Video')
    for tag in soup.find_all('oppia-noninteractive-link'):
        if not tag.has_attr('text-with-value'):
            raise utils.ValidationError("Link tag does not have 'text-with-value' attribute.")
        _raise_validation_errors_for_unescaped_html_tag(tag, 'url-with-value', 'Link')
        url = tag['url-with-value'].replace('&quot;', '').replace(' ', '')
        if utils.get_url_scheme(url) not in constants.ACCEPTABLE_SCHEMES:
            raise utils.ValidationError(f'Link should be prefix with acceptable schemas which are {constants.ACCEPTABLE_SCHEMES}')
    for tag in soup.find_all('oppia-noninteractive-math'):
        if not tag.has_attr('math_content-with-value'):
            raise utils.ValidationError("Math tag does not have 'math_content-with-value' attribute.")
        if is_html_empty(tag['math_content-with-value']):
            raise utils.ValidationError("Math tag 'math_content-with-value' attribute should not be empty.")
        math_content_json = utils.unescape_html(tag['math_content-with-value'])
        math_content_list = json.loads(math_content_json)
        if 'raw_latex' not in math_content_list:
            raise utils.ValidationError("Math tag does not have 'raw_latex-with-value' attribute.")
        if is_html_empty(math_content_list['raw_latex']):
            raise utils.ValidationError("Math tag 'raw_latex-with-value' attribute should not be empty.")
        if 'svg_filename' not in math_content_list:
            raise utils.ValidationError("Math tag does not have 'svg_filename-with-value' attribute.")
        if is_html_empty(math_content_list['svg_filename']):
            raise utils.ValidationError("Math tag 'svg_filename-with-value' attribute should not be empty.")
        if math_content_list['svg_filename'].strip()[-4:] != '.svg':
            raise utils.ValidationError("Math tag 'svg_filename-with-value' attribute should have svg extension.")
    if is_tag_nested_inside_tabs_or_collapsible:
        tabs_tags = soup.find_all('oppia-noninteractive-tabs')
        if len(tabs_tags) > 0:
            raise utils.ValidationError('Tabs tag should not be present inside another Tabs or Collapsible tag.')
        collapsible_tags = soup.find_all('oppia-noninteractive-collapsible')
        if len(collapsible_tags) > 0:
            raise utils.ValidationError('Collapsible tag should not be present inside another Tabs or Collapsible tag.')

def _raise_validation_errors_for_empty_tabs_content(content_dict: Dict[str, str], name: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Raises error when the content inside the tabs tag is empty.\n\n    Args:\n        content_dict: Dict[str]. The dictionary containing the content of\n            tags tag.\n        name: str. The content name that needs to be validated.\n\n    Raises:\n        ValidationError. Content not present in the dictionary.\n        ValidationError. Content inside the dictionary is empty.\n    '
    if name not in content_dict:
        raise utils.ValidationError('No %s attribute is present inside the tabs tag.' % name)
    if is_html_empty(content_dict[name]):
        raise utils.ValidationError('%s present inside tabs tag is empty.' % name)

def validate_tabs_and_collapsible_rte_tags(html_data: str) -> None:
    if False:
        return 10
    'Validates `Tabs` and `Collapsible` RTE tags\n\n    Args:\n        html_data: str. The RTE content of the state.\n\n    Raises:\n        ValidationError. No tabs present inside the tab_contents attribute.\n        ValidationError. No title present inside the tab_contents attribute.\n        ValidationError. Title inside the tag is empty.\n        ValidationError. No content present inside the tab_contents attribute.\n        ValidationError. Content inside the tag is empty.\n        ValidationError. No content attributes present inside the tabs tag.\n        ValidationError. No collapsible content is present inside the tag.\n        ValidationError. Collapsible content-with-value attribute is not\n            present.\n        ValidationError. Collapsible heading-with-value attribute is not\n            present.\n        ValidationError. Collapsible heading-with-value attribute is empty.\n    '
    soup = bs4.BeautifulSoup(html_data, 'html.parser')
    tabs_tags = soup.find_all('oppia-noninteractive-tabs')
    for tag in tabs_tags:
        if not tag.has_attr('tab_contents-with-value'):
            raise utils.ValidationError('No content attribute is present inside the tabs tag.')
        tab_content_json = utils.unescape_html(tag['tab_contents-with-value'])
        tab_content_list = json.loads(tab_content_json)
        if len(tab_content_list) == 0:
            raise utils.ValidationError('No tabs are present inside the tabs tag.')
        for tab_content in tab_content_list:
            _raise_validation_errors_for_empty_tabs_content(tab_content, 'title')
            _raise_validation_errors_for_empty_tabs_content(tab_content, 'content')
            validate_rte_tags(tab_content['content'], is_tag_nested_inside_tabs_or_collapsible=True)
    collapsibles_tags = soup.find_all('oppia-noninteractive-collapsible')
    for tag in collapsibles_tags:
        if not tag.has_attr('content-with-value'):
            raise utils.ValidationError('No content attribute present in collapsible tag.')
        collapsible_content_json = utils.unescape_html(tag['content-with-value'])
        collapsible_content = json.loads(collapsible_content_json).replace('\\"', '')
        if is_html_empty(collapsible_content):
            raise utils.ValidationError('No collapsible content is present inside the tag.')
        validate_rte_tags(collapsible_content, is_tag_nested_inside_tabs_or_collapsible=True)
        if not tag.has_attr('heading-with-value'):
            raise utils.ValidationError('No heading attribute present in collapsible tag.')
        collapsible_heading_json = utils.unescape_html(tag['heading-with-value'])
        collapsible_heading = json.loads(collapsible_heading_json).replace('\\"', '')
        if is_html_empty(collapsible_heading):
            raise utils.ValidationError('Heading attribute inside the collapsible tag is empty.')