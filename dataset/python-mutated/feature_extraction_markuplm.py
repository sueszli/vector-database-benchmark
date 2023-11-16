"""
Feature extractor class for MarkupLM.
"""
import html
from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...utils import is_bs4_available, logging, requires_backends
if is_bs4_available():
    import bs4
    from bs4 import BeautifulSoup
logger = logging.get_logger(__name__)

class MarkupLMFeatureExtractor(FeatureExtractionMixin):
    """
    Constructs a MarkupLM feature extractor. This can be used to get a list of nodes and corresponding xpaths from HTML
    strings.

    This feature extractor inherits from [`~feature_extraction_utils.PreTrainedFeatureExtractor`] which contains most
    of the main methods. Users should refer to this superclass for more information regarding those methods.

    """

    def __init__(self, **kwargs):
        if False:
            return 10
        requires_backends(self, ['bs4'])
        super().__init__(**kwargs)

    def xpath_soup(self, element):
        if False:
            for i in range(10):
                print('nop')
        xpath_tags = []
        xpath_subscripts = []
        child = element if element.name else element.parent
        for parent in child.parents:
            siblings = parent.find_all(child.name, recursive=False)
            xpath_tags.append(child.name)
            xpath_subscripts.append(0 if 1 == len(siblings) else next((i for (i, s) in enumerate(siblings, 1) if s is child)))
            child = parent
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        return (xpath_tags, xpath_subscripts)

    def get_three_from_single(self, html_string):
        if False:
            i = 10
            return i + 15
        html_code = BeautifulSoup(html_string, 'html.parser')
        all_doc_strings = []
        string2xtag_seq = []
        string2xsubs_seq = []
        for element in html_code.descendants:
            if type(element) == bs4.element.NavigableString:
                if type(element.parent) != bs4.element.Tag:
                    continue
                text_in_this_tag = html.unescape(element).strip()
                if not text_in_this_tag:
                    continue
                all_doc_strings.append(text_in_this_tag)
                (xpath_tags, xpath_subscripts) = self.xpath_soup(element)
                string2xtag_seq.append(xpath_tags)
                string2xsubs_seq.append(xpath_subscripts)
        if len(all_doc_strings) != len(string2xtag_seq):
            raise ValueError('Number of doc strings and xtags does not correspond')
        if len(all_doc_strings) != len(string2xsubs_seq):
            raise ValueError('Number of doc strings and xsubs does not correspond')
        return (all_doc_strings, string2xtag_seq, string2xsubs_seq)

    def construct_xpath(self, xpath_tags, xpath_subscripts):
        if False:
            print('Hello World!')
        xpath = ''
        for (tagname, subs) in zip(xpath_tags, xpath_subscripts):
            xpath += f'/{tagname}'
            if subs != 0:
                xpath += f'[{subs}]'
        return xpath

    def __call__(self, html_strings) -> BatchFeature:
        if False:
            i = 10
            return i + 15
        '\n        Main method to prepare for the model one or several HTML strings.\n\n        Args:\n            html_strings (`str`, `List[str]`):\n                The HTML string or batch of HTML strings from which to extract nodes and corresponding xpaths.\n\n        Returns:\n            [`BatchFeature`]: A [`BatchFeature`] with the following fields:\n\n            - **nodes** -- Nodes.\n            - **xpaths** -- Corresponding xpaths.\n\n        Examples:\n\n        ```python\n        >>> from transformers import MarkupLMFeatureExtractor\n\n        >>> page_name_1 = "page1.html"\n        >>> page_name_2 = "page2.html"\n        >>> page_name_3 = "page3.html"\n\n        >>> with open(page_name_1) as f:\n        ...     single_html_string = f.read()\n\n        >>> feature_extractor = MarkupLMFeatureExtractor()\n\n        >>> # single example\n        >>> encoding = feature_extractor(single_html_string)\n        >>> print(encoding.keys())\n        >>> # dict_keys([\'nodes\', \'xpaths\'])\n\n        >>> # batched example\n\n        >>> multi_html_strings = []\n\n        >>> with open(page_name_2) as f:\n        ...     multi_html_strings.append(f.read())\n        >>> with open(page_name_3) as f:\n        ...     multi_html_strings.append(f.read())\n\n        >>> encoding = feature_extractor(multi_html_strings)\n        >>> print(encoding.keys())\n        >>> # dict_keys([\'nodes\', \'xpaths\'])\n        ```'
        valid_strings = False
        if isinstance(html_strings, str):
            valid_strings = True
        elif isinstance(html_strings, (list, tuple)):
            if len(html_strings) == 0 or isinstance(html_strings[0], str):
                valid_strings = True
        if not valid_strings:
            raise ValueError(f'HTML strings must of type `str`, `List[str]` (batch of examples), but is of type {type(html_strings)}.')
        is_batched = bool(isinstance(html_strings, (list, tuple)) and isinstance(html_strings[0], str))
        if not is_batched:
            html_strings = [html_strings]
        nodes = []
        xpaths = []
        for html_string in html_strings:
            (all_doc_strings, string2xtag_seq, string2xsubs_seq) = self.get_three_from_single(html_string)
            nodes.append(all_doc_strings)
            xpath_strings = []
            for (node, tag_list, sub_list) in zip(all_doc_strings, string2xtag_seq, string2xsubs_seq):
                xpath_string = self.construct_xpath(tag_list, sub_list)
                xpath_strings.append(xpath_string)
            xpaths.append(xpath_strings)
        data = {'nodes': nodes, 'xpaths': xpaths}
        encoded_inputs = BatchFeature(data=data, tensor_type=None)
        return encoded_inputs