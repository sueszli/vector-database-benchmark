import codecs
import logging
import re
from typing import TYPE_CHECKING, Callable, Dict, Generator, Iterable, List, Optional, Set, Union, cast
if TYPE_CHECKING:
    from lxml import etree
logger = logging.getLogger(__name__)
_charset_match = re.compile(b'<\\s*meta[^>]*charset\\s*=\\s*"?([a-z0-9_-]+)"?', flags=re.I)
_xml_encoding_match = re.compile(b'\\s*<\\s*\\?\\s*xml[^>]*encoding="([a-z0-9_-]+)"', flags=re.I)
_content_type_match = re.compile('.*; *charset="?(.*?)"?(;|$)', flags=re.I)
ARIA_ROLES_TO_IGNORE = {'directory', 'menu', 'menubar', 'toolbar'}

def _normalise_encoding(encoding: str) -> Optional[str]:
    if False:
        print('Hello World!')
    "Use the Python codec's name as the normalised entry."
    try:
        return codecs.lookup(encoding).name
    except LookupError:
        return None

def _get_html_media_encodings(body: bytes, content_type: Optional[str]) -> Iterable[str]:
    if False:
        print('Hello World!')
    "\n    Get potential encoding of the body based on the (presumably) HTML body or the content-type header.\n\n    The precedence used for finding a character encoding is:\n\n    1. <meta> tag with a charset declared.\n    2. The XML document's character encoding attribute.\n    3. The Content-Type header.\n    4. Fallback to utf-8.\n    5. Fallback to windows-1252.\n\n    This roughly follows the algorithm used by BeautifulSoup's bs4.dammit.EncodingDetector.\n\n    Args:\n        body: The HTML document, as bytes.\n        content_type: The Content-Type header.\n\n    Returns:\n        The character encoding of the body, as a string.\n    "
    attempted_encodings: Set[str] = set()
    body_start = body[:1024]
    match = _charset_match.search(body_start)
    if match:
        encoding = _normalise_encoding(match.group(1).decode('ascii'))
        if encoding:
            attempted_encodings.add(encoding)
            yield encoding
    match = _xml_encoding_match.match(body_start)
    if match:
        encoding = _normalise_encoding(match.group(1).decode('ascii'))
        if encoding and encoding not in attempted_encodings:
            attempted_encodings.add(encoding)
            yield encoding
    if content_type:
        content_match = _content_type_match.match(content_type)
        if content_match:
            encoding = _normalise_encoding(content_match.group(1))
            if encoding and encoding not in attempted_encodings:
                attempted_encodings.add(encoding)
                yield encoding
    for fallback in ('utf-8', 'cp1252'):
        if fallback not in attempted_encodings:
            yield fallback

def decode_body(body: bytes, uri: str, content_type: Optional[str]=None) -> Optional['etree._Element']:
    if False:
        i = 10
        return i + 15
    '\n    This uses lxml to parse the HTML document.\n\n    Args:\n        body: The HTML document, as bytes.\n        uri: The URI used to download the body.\n        content_type: The Content-Type header.\n\n    Returns:\n        The parsed HTML body, or None if an error occurred during processed.\n    '
    if not body:
        return None
    for encoding in _get_html_media_encodings(body, content_type):
        try:
            body.decode(encoding)
        except Exception:
            pass
        else:
            break
    else:
        logger.warning('Unable to decode HTML body for %s', uri)
        return None
    from lxml import etree
    parser = etree.HTMLParser(recover=True, encoding=encoding)
    return etree.fromstring(body, parser)

def _get_meta_tags(tree: 'etree._Element', property: str, prefix: str, property_mapper: Optional[Callable[[str], Optional[str]]]=None) -> Dict[str, Optional[str]]:
    if False:
        i = 10
        return i + 15
    '\n    Search for meta tags prefixed with a particular string.\n\n    Args:\n        tree: The parsed HTML document.\n        property: The name of the property which contains the tag name, e.g.\n            "property" for Open Graph.\n        prefix: The prefix on the property to search for, e.g. "og" for Open Graph.\n        property_mapper: An optional callable to map the property to the Open Graph\n            form. Can return None for a key to ignore that key.\n\n    Returns:\n        A map of tag name to value.\n    '
    results: Dict[str, Optional[str]] = {}
    for tag in cast(List['etree._Element'], tree.xpath(f"//*/meta[starts-with(@{property}, '{prefix}:')][@content][not(@content='')]")):
        if len(results) >= 50:
            logger.warning("Skipping parsing of Open Graph for page with too many '%s:' tags", prefix)
            return {}
        key = cast(str, tag.attrib[property])
        if property_mapper:
            new_key = property_mapper(key)
            if new_key is None:
                continue
            key = new_key
        results[key] = cast(str, tag.attrib['content'])
    return results

def _map_twitter_to_open_graph(key: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    '\n    Map a Twitter card property to the analogous Open Graph property.\n\n    Args:\n        key: The Twitter card property (starts with "twitter:").\n\n    Returns:\n        The Open Graph property (starts with "og:") or None to have this property\n        be ignored.\n    '
    if key == 'twitter:card' or key == 'twitter:creator':
        return None
    if key == 'twitter:site':
        return 'og:site_name'
    return 'og' + key[7:]

def parse_html_to_open_graph(tree: 'etree._Element') -> Dict[str, Optional[str]]:
    if False:
        print('Hello World!')
    '\n    Parse the HTML document into an Open Graph response.\n\n    This uses lxml to search the HTML document for Open Graph data (or\n    synthesizes it from the document).\n\n    Args:\n        tree: The parsed HTML document.\n\n    Returns:\n        The Open Graph response as a dictionary.\n    '
    og = _get_meta_tags(tree, 'property', 'og')
    twitter = _get_meta_tags(tree, 'name', 'twitter', _map_twitter_to_open_graph)
    for (key, value) in twitter.items():
        if key not in og:
            og[key] = value
    if 'og:title' not in og:
        title = cast(List['etree._ElementUnicodeResult'], tree.xpath('((//title)[1] | (//h1)[1] | (//h2)[1] | (//h3)[1])/text()'))
        if title:
            og['og:title'] = title[0].strip()
        else:
            og['og:title'] = None
    if 'og:image' not in og:
        meta_image = cast(List['etree._ElementUnicodeResult'], tree.xpath("//*/meta[translate(@itemprop, 'IMAGE', 'image')='image'][not(@content='')]/@content[1]"))
        if meta_image:
            og['og:image'] = meta_image[0]
        else:
            images = cast(List['etree._Element'], tree.xpath('//img[@src][number(@width)>10][number(@height)>10]'))
            images = sorted(images, key=lambda i: -1 * float(i.attrib['width']) * float(i.attrib['height']))
            if not images:
                images = cast(List['etree._Element'], tree.xpath('//img[@src][1]'))
            if images:
                og['og:image'] = cast(str, images[0].attrib['src'])
            else:
                favicons = cast(List['etree._ElementUnicodeResult'], tree.xpath("//link[@href][contains(@rel, 'icon')]/@href[1]"))
                if favicons:
                    og['og:image'] = favicons[0]
    if 'og:description' not in og:
        meta_description = cast(List['etree._ElementUnicodeResult'], tree.xpath("//*/meta[translate(@name, 'DESCRIPTION', 'description')='description'][not(@content='')]/@content[1]"))
        if meta_description:
            og['og:description'] = meta_description[0]
        else:
            og['og:description'] = parse_html_description(tree)
    elif og['og:description']:
        assert isinstance(og['og:description'], str)
        og['og:description'] = summarize_paragraphs([og['og:description']])
    return og

def parse_html_description(tree: 'etree._Element') -> Optional[str]:
    if False:
        i = 10
        return i + 15
    '\n    Calculate a text description based on an HTML document.\n\n    Grabs any text nodes which are inside the <body/> tag, unless they are within\n    an HTML5 semantic markup tag (<header/>, <nav/>, <aside/>, <footer/>), or\n    if they are within a <script/>, <svg/> or <style/> tag, or if they are within\n    a tag whose content is usually only shown to old browsers\n    (<iframe/>, <video/>, <canvas/>, <picture/>).\n\n    This is a very very very coarse approximation to a plain text render of the page.\n\n    Args:\n        tree: The parsed HTML document.\n\n    Returns:\n        The plain text description, or None if one cannot be generated.\n    '
    from lxml import etree
    TAGS_TO_REMOVE = {'header', 'nav', 'aside', 'footer', 'script', 'noscript', 'style', 'svg', 'iframe', 'video', 'canvas', 'img', 'picture', etree.Comment}
    text_nodes = (re.sub('\\s+', '\n', el).strip() for el in _iterate_over_text(tree.find('body'), TAGS_TO_REMOVE))
    return summarize_paragraphs(text_nodes)

def _iterate_over_text(tree: Optional['etree._Element'], tags_to_ignore: Set[object], stack_limit: int=1024) -> Generator[str, None, None]:
    if False:
        return 10
    "Iterate over the tree returning text nodes in a depth first fashion,\n    skipping text nodes inside certain tags.\n\n    Args:\n        tree: The parent element to iterate. Can be None if there isn't one.\n        tags_to_ignore: Set of tags to ignore\n        stack_limit: Maximum stack size limit for depth-first traversal.\n            Nodes will be dropped if this limit is hit, which may truncate the\n            textual result.\n            Intended to limit the maximum working memory when generating a preview.\n    "
    if tree is None:
        return
    elements: List[Union[str, 'etree._Element']] = [tree]
    while elements:
        el = elements.pop()
        if isinstance(el, str):
            yield el
        elif el.tag not in tags_to_ignore:
            if el.get('role') in ARIA_ROLES_TO_IGNORE:
                continue
            if el.text:
                yield el.text
            for child in el.iterchildren(reversed=True):
                if len(elements) > stack_limit:
                    break
                if child.tail:
                    elements.append(child.tail)
                elements.append(child)

def summarize_paragraphs(text_nodes: Iterable[str], min_size: int=200, max_size: int=500) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Try to get a summary respecting first paragraph and then word boundaries.\n\n    Args:\n        text_nodes: The paragraphs to summarize.\n        min_size: The minimum number of words to include.\n        max_size: The maximum number of words to include.\n\n    Returns:\n        A summary of the text nodes, or None if that was not possible.\n    '
    description = ''
    for text_node in text_nodes:
        if len(description) < min_size:
            text_node = re.sub('[\\t \\r\\n]+', ' ', text_node)
            description += text_node + '\n\n'
        else:
            break
    description = description.strip()
    description = re.sub('[\\t ]+', ' ', description)
    description = re.sub('[\\t \\r\\n]*[\\r\\n]+', '\n\n', description)
    if len(description) > max_size:
        new_desc = ''
        for match in re.finditer('\\s*\\S+', description):
            word = match.group()
            if len(word) + len(new_desc) < max_size:
                new_desc += word
            else:
                if len(new_desc) < min_size:
                    new_desc += word
                break
        if len(new_desc) > max_size:
            new_desc = new_desc[:max_size]
        description = new_desc.strip() + '…'
    return description if description else None